from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

APP_SUPPORT_DIR = Path.home() / "Library" / "Application Support" / "VoiceFlow"
CONFIG_PATH = APP_SUPPORT_DIR / "config.json"

DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_MAX_ACCURACY_MODEL = "mlx-community/whisper-large-v3-mlx"
_INVALID_MODEL_ALIASES = {
    "mlx-community/whisper-large-v3",
}


@dataclass
class AppConfig:
    recording_mode: str = "push_to_talk"
    hotkey: str = "right_cmd"
    silence_duration_ms: int = 700
    vad_threshold: float = 0.5
    whisper_model: str = DEFAULT_WHISPER_MODEL
    language: str = "en"
    cleanup_mode: str = "standard"
    llm_model: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    restore_clipboard: bool = True
    dictionary_path: str = ""
    max_accuracy_whisper_model: str = DEFAULT_MAX_ACCURACY_MODEL

    def __post_init__(self) -> None:
        if not self.dictionary_path:
            self.dictionary_path = str(APP_SUPPORT_DIR / "dictionary.json")
        if not self.whisper_model or self.whisper_model in _INVALID_MODEL_ALIASES:
            self.whisper_model = DEFAULT_WHISPER_MODEL
        if (
            not self.max_accuracy_whisper_model
            or self.max_accuracy_whisper_model in _INVALID_MODEL_ALIASES
        ):
            self.max_accuracy_whisper_model = DEFAULT_MAX_ACCURACY_MODEL

    @classmethod
    def load(cls) -> AppConfig:
        APP_SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text())
                known_fields = {f.name for f in cls.__dataclass_fields__.values()}
                filtered = {k: v for k, v in data.items() if k in known_fields}
                config = cls(**filtered)
                # Persist automatic migrations (e.g., deprecated model IDs).
                if filtered.get("whisper_model") != config.whisper_model or filtered.get(
                    "max_accuracy_whisper_model"
                ) != config.max_accuracy_whisper_model:
                    config.save()
                return config
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                log.warning("Corrupted config file, using defaults: %s", e)
        config = cls()
        config.save()
        return config

    def save(self) -> None:
        APP_SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2) + "\n")
