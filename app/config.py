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
DEFAULT_LLM_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
DEFAULT_LANGUAGE = "auto"
_INVALID_MODEL_ALIASES = {
    "mlx-community/whisper-large-v3",
}
_LLM_MODEL_ALIASES = {
    # Requested 4B Minitron variants are not published on MLX; map to stable <4B.
    "mlx-community/Mistral-NeMo-Minitron-4B-Instruct": DEFAULT_LLM_MODEL,
    "mlx-community/Mistral-NeMo-Minitron-4B-Instruct-4bit": DEFAULT_LLM_MODEL,
    "nvidia/Mistral-NeMo-Minitron-4B-Instruct": DEFAULT_LLM_MODEL,
    "mlx-community/Mistral-NeMo-Minitron-8B-Instruct-4bit": DEFAULT_LLM_MODEL,
}
_LANGUAGE_ALIASES = {
    "en": "en",
    "english": "en",
    "de": "de",
    "deutsch": "de",
    "german": "de",
    "auto": "auto",
    "multilingual": "auto",
    "english_german": "auto",
    "en_de": "auto",
}
_TRANSCRIPTION_MODE_ALIASES = {
    "normal": "normal",
    "general": "normal",
    "default": "normal",
    "programmer": "programmer",
    "coding": "programmer",
    "developer": "programmer",
}
_DEFAULT_PROGRAMMER_APPS = [
    "codex",
    "claude",
    "terminal",
    "iterm",
    "warp",
    "cursor",
    "visual studio code",
    "vscode",
    "code",
    "xcode",
    "zed",
    "sublime text",
    "jetbrains",
    "pycharm",
    "webstorm",
    "intellij",
]


@dataclass
class AppConfig:
    transcription_mode: str = "programmer"
    auto_mode_switch: bool = True
    programmer_apps: list[str] = field(default_factory=lambda: list(_DEFAULT_PROGRAMMER_APPS))
    recording_mode: str = "push_to_talk"
    hotkey: str = "right_cmd"
    silence_duration_ms: int = 700
    vad_threshold: float = 0.5
    whisper_model: str = DEFAULT_WHISPER_MODEL
    language: str = DEFAULT_LANGUAGE
    cleanup_mode: str = "standard"
    llm_model: str = DEFAULT_LLM_MODEL
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
        self.language = _LANGUAGE_ALIASES.get(
            str(self.language).strip().lower(), DEFAULT_LANGUAGE
        )
        self.transcription_mode = _TRANSCRIPTION_MODE_ALIASES.get(
            str(self.transcription_mode).strip().lower(),
            "programmer",
        )
        self.auto_mode_switch = bool(self.auto_mode_switch)
        if isinstance(self.programmer_apps, str):
            raw_apps = self.programmer_apps.split(",")
        elif isinstance(self.programmer_apps, list):
            raw_apps = self.programmer_apps
        else:
            raw_apps = []
        cleaned_apps = [
            str(item).strip().lower() for item in raw_apps if str(item).strip()
        ]
        self.programmer_apps = cleaned_apps or list(_DEFAULT_PROGRAMMER_APPS)
        if not self.llm_model:
            self.llm_model = DEFAULT_LLM_MODEL
        self.llm_model = _LLM_MODEL_ALIASES.get(self.llm_model, self.llm_model)

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
                if (
                    filtered.get("whisper_model") != config.whisper_model
                    or filtered.get("max_accuracy_whisper_model")
                    != config.max_accuracy_whisper_model
                    or filtered.get("llm_model") != config.llm_model
                    or filtered.get("transcription_mode")
                    != config.transcription_mode
                    or filtered.get("auto_mode_switch") != config.auto_mode_switch
                    or filtered.get("programmer_apps") != config.programmer_apps
                ):
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
