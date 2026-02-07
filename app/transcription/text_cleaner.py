from __future__ import annotations

import re


_FILLER_REMOVE: list[re.Pattern[str]] = [
    re.compile(r'\b(um+|uh+|hmm+|hm+|ah+|eh+|er+|oh+)\b', re.IGNORECASE),
    re.compile(r'\b(so yeah|and yeah|yeah so|right so)\b[.,]?', re.IGNORECASE),
]

_FILLER_REPLACE_SPACE = re.compile(
    r',?\s*\b(you know|I mean|sort of|kind of|basically|actually|literally)\b\s*,?',
    re.IGNORECASE,
)

_REPEATED_WORD = re.compile(r'\b(\w+)(\s+\1)+\b', re.IGNORECASE)


class TextCleaner:
    @classmethod
    def clean(cls, text: str, dictionary: dict[str, str] | None = None) -> str:
        for pattern in _FILLER_REMOVE:
            text = pattern.sub('', text)
        text = _FILLER_REPLACE_SPACE.sub(' ', text)
        text = _REPEATED_WORD.sub(r'\1', text)

        if dictionary:
            for wrong, right in sorted(dictionary.items(), key=lambda kv: -len(kv[0])):
                text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)

        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'^[,\s]+', '', text)
        return text.strip()
