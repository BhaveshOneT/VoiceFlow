from __future__ import annotations

import re


_FILLER_REMOVE: list[re.Pattern[str]] = [
    re.compile(r'\b(um+|uh+|hmm+|hm+|ah+|eh+|er+|oh+)\b', re.IGNORECASE),
    re.compile(r'\b(so yeah|and yeah|yeah so|right so)\b[.,]?', re.IGNORECASE),
]

_FILLER_REPLACE_SPACE = re.compile(
    r',?\s*\b(you know|sort of|kind of|basically|literally)\b\s*,?',
    re.IGNORECASE,
)

_REPEATED_WORD = re.compile(r'\b(\w+)(\s+\1)+\b', re.IGNORECASE)

_LEADING_DISCOURSE = re.compile(
    r'^\s*(?:(?:okay|ok|well|so)\s*,?\s*)+',
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
_CORRECTION_PREFIX = re.compile(
    r'^\s*(?:no(?:\s*,\s*no)?|sorry|actually|rather|correction|'
    r'i mean|i meant|wait no|no wait)\b[\s,:-]*',
    re.IGNORECASE,
)
_INLINE_CORRECTION = re.compile(
    r'^(?P<prefix>.+?)\s*(?:,\s*|\s+)'
    r'(?P<cue>sorry|actually|rather|i mean|i meant|no wait|wait no|no\s*,?\s*no)\b'
    r'[\s,:-]*(?P<replacement>.+)$',
    re.IGNORECASE,
)
_VERB_TARGET_OF_APP = re.compile(
    r'^(.*?\b(?:change|update|modify|refactor|improve|fix)\b\s+)'
    r'(?:the\s+)?(.+?)'
    r'(\s+of\s+the\s+app)([.!?]?)$',
    re.IGNORECASE,
)
_VERB_TO_TARGET = re.compile(
    r'^(.*?\b(?:change|set|switch|rename|call|use|move)\b\s+'
    r'(?:it|this|that|the\s+\w+)?\s*to\s+)'
    r'(.+?)([.!?]?)$',
    re.IGNORECASE,
)
_VERB_TRAILING_TOKEN = re.compile(
    r'^(.*?\b(?:call|name|rename|select|choose)\b\s+'
    r'(?:the\s+\w+\s+)?)'
    r'([A-Za-z0-9_.:-]+)([.!?]?)$',
    re.IGNORECASE,
)
_VERB_OPEN_END = re.compile(
    r'^(.*?\b(?:use|call|name|rename|set|switch|move)\b)\s*$',
    re.IGNORECASE,
)
_FILE_EXTS = (
    "py",
    "js",
    "jsx",
    "ts",
    "tsx",
    "java",
    "go",
    "rs",
    "rb",
    "php",
    "swift",
    "kt",
    "c",
    "h",
    "hpp",
    "cpp",
    "m",
    "mm",
    "cs",
    "json",
    "yaml",
    "yml",
    "toml",
    "ini",
    "env",
    "md",
    "txt",
    "sql",
    "sh",
    "bash",
    "zsh",
    "html",
    "htm",
    "css",
    "scss",
    "vue",
)
_FILE_EXT_ALT = "|".join(_FILE_EXTS)
_EXPLICIT_FILE_RE = re.compile(
    rf'(?<![\w@])(?P<name>[A-Za-z0-9][A-Za-z0-9_./-]*\.(?:{_FILE_EXT_ALT}))\b'
    r'(?:\s+file\b)?',
    re.IGNORECASE,
)
_SPOKEN_DOT_FILE_RE = re.compile(
    rf'(?<![\w@])(?P<base>[A-Za-z0-9][A-Za-z0-9_-]*)\s+dot\s+'
    rf'(?P<ext>{_FILE_EXT_ALT})\b(?:\s+file\b)?',
    re.IGNORECASE,
)
_DUPLICATE_FILE_TAG_RE = re.compile(r'@\s+@\s+')
_BARE_FILE_START_BLOCK = (
    "a|an|the|this|that|my|your|our|their|open|close|read|write|save|edit|"
    "modify|update|change|fix|move|rename|create|delete|remove|use|call|set|"
    "switch|want|need|have|is|are|was|were|please|just|to"
)
_BARE_FILE_RE = re.compile(
    rf'(?<![@\w])(?P<base>(?!(?:{_BARE_FILE_START_BLOCK})\b)'
    r'[A-Za-z][A-Za-z0-9_-]*(?:\s+[A-Za-z0-9_-]+)?)\s+file\b',
    re.IGNORECASE,
)
_GENERIC_FILE_BASES = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "it",
    "my",
    "your",
    "our",
    "their",
}
_ACTION_CLAUSE_RE = re.compile(
    r'^(?P<head>.*?)(?P<clause>(?:i\s+(?:want|need)\s+to\s+)?'
    r'(?:change|update|modify|refactor|improve|fix|rename|move|set|switch|use|call)\b.+)$',
    re.IGNORECASE,
)
_INTENT_PREFIX_RE = re.compile(
    r'^(?P<intent>i\s+(?:want|need)\s+to)\s+(?P<rest>.+)$',
    re.IGNORECASE,
)
_ACTION_START_RE = re.compile(
    r'^(?:i\s+(?:want|need)\s+to\s+)?'
    r'(?:change|update|modify|refactor|improve|fix|rename|move|set|switch|use|call)\b',
    re.IGNORECASE,
)
_CLAUSE_SPLIT_RE = re.compile(r'(?<=[.!?;:])\s+')


class TextCleaner:
    @classmethod
    def clean(cls, text: str, dictionary: dict[str, str] | None = None) -> str:
        for pattern in _FILLER_REMOVE:
            text = pattern.sub('', text)
        text = _FILLER_REPLACE_SPACE.sub(' ', text)
        text = _REPEATED_WORD.sub(cls._dedupe_repeated_word, text)

        if dictionary:
            for wrong, right in sorted(dictionary.items(), key=lambda kv: -len(kv[0])):
                text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)

        text = cls._apply_self_corrections(text)
        text = cls._collapse_repeated_clauses(text)
        text = cls._tag_file_mentions(text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'^[,\s]+', '', text)
        return text.strip()

    @staticmethod
    def _dedupe_repeated_word(match: re.Match[str]) -> str:
        token = match.group(1)
        # Keep "no no" for explicit self-correction detection.
        if token.lower() == "no":
            return match.group(0)
        return token

    @classmethod
    def _apply_self_corrections(cls, text: str) -> str:
        """Rewrite explicit backtracks such as 'no, no, X' using prior context."""
        sentences = _SENTENCE_SPLIT.split(text.strip())
        out: list[str] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            inline = _INLINE_CORRECTION.match(sentence)
            if inline:
                prefix = inline.group("prefix").strip()
                replacement = inline.group("replacement").strip(" ,.-")
                out.append(cls._merge_with_previous(prefix, replacement))
                continue
            match = _CORRECTION_PREFIX.match(sentence)
            if match:
                replacement = sentence[match.end():].strip(" ,.-")
                if not replacement:
                    continue
                if out:
                    out[-1] = cls._merge_with_previous(out[-1], replacement)
                else:
                    out.append(replacement)
                continue
            out.append(sentence)
        return " ".join(out)

    @classmethod
    def _merge_with_previous(cls, previous: str, replacement: str) -> str:
        previous = _LEADING_DISCOURSE.sub("", previous).strip()
        replacement = _LEADING_DISCOURSE.sub("", replacement).strip()
        replacement = replacement.rstrip(".!?")
        while True:
            stripped = _CORRECTION_PREFIX.sub("", replacement).strip(" ,.-")
            if stripped == replacement:
                break
            replacement = stripped

        # Pattern: "... change the functionality of the app." + "modularity of the app"
        match = _VERB_TARGET_OF_APP.match(previous)
        if match:
            prefix, _, suffix, punctuation = match.groups()
            rep = replacement
            if rep.lower().endswith("of the app"):
                rep = rep[:-len("of the app")].strip()
            if rep:
                article = "" if re.match(r"^(the|a|an)\b", rep, re.IGNORECASE) else "the "
                punct = punctuation or "."
                return f"{prefix}{article}{rep}{suffix}{punct}"

        # Pattern: "... change it to X" + "Y"
        match = _VERB_TO_TARGET.match(previous)
        if match and replacement:
            prefix, _, punctuation = match.groups()
            punct = punctuation or "."
            return f"{prefix}{replacement}{punct}"

        match = _VERB_TRAILING_TOKEN.match(previous)
        if match and replacement:
            prefix, _, punctuation = match.groups()
            punct = punctuation or "."
            return f"{prefix}{replacement}{punct}"

        match = _VERB_OPEN_END.match(previous)
        if match and replacement:
            prefix = match.group(1).strip()
            return f"{prefix} {replacement}."

        # Preserve context for action-style corrections:
        # "we have a problem ... I want to modify X no no modify Y"
        action_match = _ACTION_CLAUSE_RE.match(previous)
        if action_match and _ACTION_START_RE.match(replacement):
            head = action_match.group("head").strip()
            clause = action_match.group("clause").strip().rstrip(".!?")
            replacement_clause = replacement.rstrip(".!?")
            intent_match = _INTENT_PREFIX_RE.match(clause)
            if intent_match and not _INTENT_PREFIX_RE.match(replacement_clause):
                replacement_clause = (
                    f"{intent_match.group('intent').strip()} {replacement_clause}"
                )
            merged = f"{head} {replacement_clause}".strip()
            return merged + "."

        # Fallback: replace previous sentence with replacement.
        return replacement + "."

    @classmethod
    def _tag_file_mentions(cls, text: str) -> str:
        """Turn spoken or explicit file mentions into @-style file tags."""
        text = _SPOKEN_DOT_FILE_RE.sub(cls._replace_spoken_file, text)
        text = _EXPLICIT_FILE_RE.sub(cls._replace_explicit_file, text)
        text = _BARE_FILE_RE.sub(cls._replace_bare_file, text)
        text = _DUPLICATE_FILE_TAG_RE.sub("@ ", text)
        return text

    @staticmethod
    def _collapse_repeated_clauses(text: str) -> str:
        """Collapse immediate repeated clauses (common ASR loop artifact)."""
        out: list[str] = []
        prev_norm = ""
        for chunk in _CLAUSE_SPLIT_RE.split(text.strip()):
            chunk = chunk.strip()
            if not chunk:
                continue
            body = chunk.rstrip(".!?;:").strip()
            if not body:
                continue
            norm = re.sub(r"\s+", " ", body).strip().lower()
            word_count = len(norm.split())
            if norm == prev_norm and word_count >= 3:
                continue
            out.append(chunk)
            prev_norm = norm
        return " ".join(out) if out else text

    @staticmethod
    def _replace_spoken_file(match: re.Match[str]) -> str:
        base = match.group("base")
        ext = match.group("ext").lower()
        return f"@ {base}.{ext}"

    @staticmethod
    def _replace_explicit_file(match: re.Match[str]) -> str:
        name = match.group("name")
        return f"@ {name}"

    @staticmethod
    def _replace_bare_file(match: re.Match[str]) -> str:
        base = match.group("base").strip()
        lowered = base.lower()
        if lowered in _GENERIC_FILE_BASES:
            return match.group(0)
        tag = re.sub(r"\s+", "_", base.strip())
        return f"@ {tag}"
