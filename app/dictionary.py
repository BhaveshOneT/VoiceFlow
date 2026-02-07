from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


RESOURCES_DIR = Path(__file__).parent / "resources"
DEFAULT_DICTIONARY_PATH = RESOURCES_DIR / "default_dictionary.json"


@dataclass
class Dictionary:
    terms: dict[str, str] = field(default_factory=dict)
    auto_learned: dict[str, str] = field(default_factory=dict)
    correction_counts: dict[str, int] = field(default_factory=dict)
    AUTO_LEARN_THRESHOLD: int = 3

    _save_path: Path | None = field(default=None, repr=False)

    @classmethod
    def load_defaults(cls) -> Dictionary:
        if DEFAULT_DICTIONARY_PATH.exists():
            try:
                data = json.loads(DEFAULT_DICTIONARY_PATH.read_text())
                return cls(terms=data.get("terms", {}))
            except (json.JSONDecodeError, TypeError) as e:
                log.warning("Corrupted default dictionary, using empty: %s", e)
        return cls()

    @classmethod
    def load(cls, path: Path) -> Dictionary:
        defaults = cls.load_defaults()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                merged_terms = {**defaults.terms, **data.get("terms", {})}
                inst = cls(
                    terms=merged_terms,
                    auto_learned=data.get("auto_learned", {}),
                    correction_counts=data.get("correction_counts", {}),
                )
            except (json.JSONDecodeError, TypeError) as e:
                log.warning("Corrupted user dictionary, using defaults: %s", e)
                inst = defaults
        else:
            inst = defaults
        inst._save_path = path
        return inst

    def save(self, path: Path | None = None) -> None:
        target = path or self._save_path
        if target is None:
            raise ValueError("No save path specified")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "terms": self.terms,
            "auto_learned": self.auto_learned,
            "correction_counts": self.correction_counts,
        }
        target.write_text(json.dumps(data, indent=2) + "\n")

    def record_correction(self, wrong: str, right: str) -> None:
        key = wrong.lower()
        self.correction_counts[key] = self.correction_counts.get(key, 0) + 1
        if self.correction_counts[key] >= self.AUTO_LEARN_THRESHOLD:
            self.auto_learned[key] = right
            if self._save_path:
                self.save()

    def get_all_terms(self) -> dict[str, str]:
        return {**self.terms, **self.auto_learned}

    def get_whisper_context(self) -> str:
        unique_values = list(dict.fromkeys(self.get_all_terms().values()))
        top_terms = unique_values[:20]
        if not top_terms:
            return ""
        first_half = top_terms[:10]
        second_half = top_terms[10:20]
        context = f"In this session, we're working with {', '.join(first_half)}."
        if second_half:
            context += f" The project also uses {', '.join(second_half)}."
        return context
