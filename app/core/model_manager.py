"""Centralized model lifecycle manager.

Ensures only one heavy ML model family is loaded at a time to prevent OOM
on 16GB Apple Silicon Macs. Coordinates between:
  - STT (Parakeet by default, Whisper-compatible fallback)
  - pyannote (diarization, ~1.5GB + PyTorch)
  - Qwen (LLM refiner/summarizer, ~2GB)
"""
from __future__ import annotations

import gc
import logging
import threading
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.signals import AppSignals

log = logging.getLogger(__name__)


class ModelSlot(Enum):
    """Which model family is currently occupying GPU memory."""
    NONE = auto()
    STT = auto()
    WHISPER = STT  # Backward-compatible alias for legacy callers.
    DIARIZATION = auto()
    SUMMARIZER = auto()


class ModelManager:
    """Coordinates model loading/unloading to stay within memory budget.

    Usage::

        mm = ModelManager(signals)
        mm.prepare_for_dictation()        # loads STT
        mm.prepare_for_diarization()      # unloads STT, loads pyannote
        mm.prepare_for_summarization()    # unloads pyannote, loads Qwen
    """

    def __init__(self, signals: "AppSignals | None" = None) -> None:
        self._signals = signals
        self._lock = threading.Lock()
        self._active_slot = ModelSlot.NONE

        # External model references (set by the owner)
        self._stt_unload = None  # callable
        self._stt_load = None  # callable
        self._diarizer_unload = None
        self._diarizer_load = None
        self._summarizer_unload = None
        self._summarizer_load = None

    @property
    def active_slot(self) -> ModelSlot:
        return self._active_slot

    def register_stt(self, load_fn, unload_fn) -> None:
        self._stt_load = load_fn
        self._stt_unload = unload_fn

    def register_whisper(self, load_fn, unload_fn) -> None:
        """Backward-compatible alias for older call sites."""
        self.register_stt(load_fn=load_fn, unload_fn=unload_fn)

    def register_diarizer(self, load_fn, unload_fn) -> None:
        self._diarizer_load = load_fn
        self._diarizer_unload = unload_fn

    def register_summarizer(self, load_fn, unload_fn) -> None:
        self._summarizer_load = load_fn
        self._summarizer_unload = unload_fn

    def prepare_for_dictation(self) -> None:
        """Ensure STT is loaded (+ optional Qwen for refiner)."""
        self._switch_to(ModelSlot.STT)

    def prepare_for_meeting_transcription(self) -> None:
        """Ensure STT is loaded for segment-level transcription."""
        self._switch_to(ModelSlot.STT)

    def prepare_for_diarization(self) -> None:
        """Unload STT, load pyannote diarization pipeline."""
        self._switch_to(ModelSlot.DIARIZATION)

    def prepare_for_summarization(self) -> None:
        """Unload everything, load Qwen for summarization."""
        self._switch_to(ModelSlot.SUMMARIZER)

    def unload_all(self) -> None:
        """Unload all models and free memory."""
        with self._lock:
            self._unload_current()
            self._active_slot = ModelSlot.NONE

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _switch_to(self, target: ModelSlot) -> None:
        with self._lock:
            if self._active_slot == target:
                return

            self._unload_current()
            self._load_slot(target)
            self._active_slot = target

    def _unload_current(self) -> None:
        slot = self._active_slot
        if slot == ModelSlot.NONE:
            return

        log.info("Unloading model slot: %s", slot.name)
        if self._signals:
            self._signals.status_changed.emit(f"Unloading {slot.name.lower()} model...")

        if slot == ModelSlot.STT and self._stt_unload:
            self._stt_unload()
        elif slot == ModelSlot.DIARIZATION and self._diarizer_unload:
            self._diarizer_unload()
        elif slot == ModelSlot.SUMMARIZER and self._summarizer_unload:
            self._summarizer_unload()

        self._free_memory()

    def _load_slot(self, slot: ModelSlot) -> None:
        log.info("Loading model slot: %s", slot.name)
        if self._signals:
            self._signals.model_loading.emit(slot.name.lower())

        if slot == ModelSlot.STT and self._stt_load:
            self._stt_load()
        elif slot == ModelSlot.DIARIZATION and self._diarizer_load:
            self._diarizer_load()
        elif slot == ModelSlot.SUMMARIZER and self._summarizer_load:
            self._summarizer_load()

        if self._signals:
            self._signals.model_loaded.emit(slot.name.lower())

    @staticmethod
    def _free_memory() -> None:
        """Aggressively free GPU memory."""
        gc.collect()
        try:
            import torch
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass
