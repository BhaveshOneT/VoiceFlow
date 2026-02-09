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
_INLINE_DISCOURSE_RE = re.compile(
    r"\b(?:we can see|you can see|we'?ll see|let'?s see)\b",
    re.IGNORECASE,
)
_HESITATION_CHAIN_RE = re.compile(
    r"\b(?:i don't know|i do not know)\s+(?:yeah\s+)?maybe\b",
    re.IGNORECASE,
)
_YEAH_FILLER_RE = re.compile(r"(?:(?<=\s)|^)(?:yeah|yep)(?=(?:\s|$|[,.!?;:]))", re.IGNORECASE)

_REPEATED_WORD = re.compile(r'\b(\w+)(\s+\1)+\b', re.IGNORECASE)

_LEADING_DISCOURSE = re.compile(
    r'^\s*(?:(?:okay|ok|well|so)\s*,?\s*)+',
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
_CORRECTION_PREFIX = re.compile(
    r'^\s*(?P<cue>no\s*,\s*no|no\s+no|sorry|rather|correction|'
    r'i mean|i meant|wait no|no wait|scratch that|never mind(?: that)?|'
    r'let me rephrase)\b[\s,:-]*',
    re.IGNORECASE,
)
_INLINE_CORRECTION = re.compile(
    r'^(?P<prefix>.+?)\s*(?:,\s*|\s+)'
    r'(?P<cue>sorry|rather|i mean|i meant|no wait|wait no|no\s*,?\s*no|'
    r'scratch that|never mind(?: that)?|let me rephrase)\b'
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
    "dmg",
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
_SPOKEN_COMPLEX_FILE_RE = re.compile(
    rf'(?<![\w@])(?P<base>[A-Za-z0-9][A-Za-z0-9_-]*'
    r'(?:\s+(?:underscore|under score|dash|hyphen)\s+[A-Za-z0-9][A-Za-z0-9_-]*)+)'
    rf'\s+dot\s+(?P<ext>{_FILE_EXT_ALT})\b(?:\s+file\b)?',
    re.IGNORECASE,
)
_DUPLICATE_FILE_TAG_RE = re.compile(r'@\s*@\s*')
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
_JS_CONTEXT_HINTS = {
    "next",
    "react",
    "node",
    "express",
    "nest",
    "vite",
    "vue",
    "nuxt",
    "remix",
    "solid",
    "plate",
}
_JS_HOMOPHONE_RE = re.compile(r"\b(?P<base>[A-Za-z][A-Za-z0-9_-]*)\s+chess\b", re.IGNORECASE)
_SPELLED_JS_RE = re.compile(r"\b(jay\s+ess|j\s*\.?\s*s)\b", re.IGNORECASE)
_SPELLED_TS_RE = re.compile(r"\b(tea\s+ess|t\s*\.?\s*s)\b", re.IGNORECASE)
_SYMBOL_MENTION_RE = re.compile(
    r"\b(?P<verb>update|modify|refactor|fix|rename|call|use|create|open|check|test)\s+"
    r"(?:the\s+)?(?P<kind>function|method|class|module|variable|interface|type)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_.:-]{1,64})\b",
    re.IGNORECASE,
)
_SYMBOL_FILE_EXT_RE = re.compile(rf"\.(?:{_FILE_EXT_ALT})$", re.IGNORECASE)
_GENERIC_SYMBOLS = {
    "code",
    "file",
    "app",
    "function",
    "class",
    "module",
    "variable",
    "type",
    "interface",
}
_DUPLICATE_SYMBOL_TAG_RE = re.compile(
    r"(@[A-Za-z_][A-Za-z0-9_.:-]*)(?:\s+\1)+"
)
_CLAUSE_SPLIT_RE = re.compile(r'(?<=[.!?;:])\s+')
_SOFT_CLAUSE_SPLIT_RE = re.compile(r'(?<=[,.!?;:])\s+')
_LOW_INFO_FRAGMENT_RE = re.compile(
    r"^(?:"
    r"okay|ok|yeah|right|you know|i mean|let'?s see|we can see|you can see|"
    r"we'?ll see|i guess|i don't know|i do not know"
    r")$",
    re.IGNORECASE,
)
_TRIM_EDGE_PUNCT_RE = re.compile(r"^[\s,;:.!?-]+|[\s,;:.!?-]+$")
_LEADING_LOWER_RE = re.compile(r"(^|(?<=[.!?]\s))([a-z])")
_I_CONTRACTION_RE = re.compile(r"\bi(?=('|’)(m|d|ll|ve|re|s)\b)", re.IGNORECASE)
_STANDALONE_I_RE = re.compile(r"\bi\b", re.IGNORECASE)
_TERMINAL_PUNCT_RE = re.compile(r'[.!?]["\')\]]?$')
_LONE_EXTENSION_TAG_RE = re.compile(rf'(?<![\w])@(?P<ext>{_FILE_EXT_ALT})\b', re.IGNORECASE)
_TRAILING_CONJUNCTION_RE = re.compile(
    r"\b(?:and|or|but|so|because|then)\b\s*$",
    re.IGNORECASE,
)
_MISSING_SENTENCE_BREAK_RE = re.compile(
    r"(?<=[a-z0-9])\s+(?=(?:The|Then|And|But)\s+[A-Z]?[a-z])"
)
_VERB_PREFIX_TAG_FILE_RE = re.compile(
    rf"\b(?P<verb>rename|update|modify|edit|open|create|delete|move|copy)\s+"
    rf"(?P<middle>(?:(?:the|this|that)\s+)?(?:file\s+)?)?"
    rf"(?P<prefix>[A-Za-z0-9_-]{{2,}})\s+@(?P<name>[A-Za-z0-9_-]+\.(?:{_FILE_EXT_ALT}))\b",
    re.IGNORECASE,
)
_FRAGMENTED_TAG_RE = re.compile(
    rf'@(?P<left>[A-Za-z0-9_-]+)(?P<sep>[-_])@(?P<right>[A-Za-z0-9_-]+\.(?:{_FILE_EXT_ALT}))\b',
    re.IGNORECASE,
)
_SPOKEN_FRAGMENTED_TAG_RE = re.compile(
    rf'(?<![@\w])(?P<left>[A-Za-z0-9_-]+)\s+'
    r'(?P<sep>underscore|under score|dash|hyphen)\s+'
    rf'@(?P<right>[A-Za-z0-9_-]+\.(?:{_FILE_EXT_ALT}))\b',
    re.IGNORECASE,
)
_FRAMEWORK_FILE_TOKENS = {
    "next.js",
    "node.js",
    "react.js",
    "plate.js",
    "vue.js",
    "nuxt.js",
    "solid.js",
    "svelte.js",
    "express.js",
}
_TAGGED_JS_LIST_RE = re.compile(
    r'(?P<prefix>\b(?:terms?|libraries|frameworks?)\s+like\s+)'
    r'(?P<body>@[A-Za-z0-9_-]+\.(?:js|jsx|ts|tsx)\b'
    r'(?:\s*,\s*@[A-Za-z0-9_-]+\.(?:js|jsx|ts|tsx)\b)*'
    r'(?:\s+and\s+@[A-Za-z0-9_-]+\.(?:js|jsx|ts|tsx)\b)?)',
    re.IGNORECASE,
)
_EMBEDDED_SHOULD_QUESTION_RE = re.compile(
    r"\bif\s+i\s+ask\s+should\s+(?P<body>.+?)\s+(?=keep it as a question\b)",
    re.IGNORECASE,
)
_STRONG_REPLACE_CUES = {
    "no no",
    "no wait",
    "wait no",
    "i mean",
    "i meant",
    "rather",
    "correction",
    "scratch that",
    "never mind",
    "never mind that",
    "let me rephrase",
}
_WEAK_REPLACE_CUES = {"sorry"}


class TextCleaner:
    @classmethod
    def clean(
        cls,
        text: str,
        dictionary: dict[str, str] | None = None,
        programmer_mode: bool = True,
    ) -> str:
        for pattern in _FILLER_REMOVE:
            text = pattern.sub('', text)
        text = _LEADING_DISCOURSE.sub('', text)
        text = _INLINE_DISCOURSE_RE.sub(' ', text)
        text = _HESITATION_CHAIN_RE.sub('maybe', text)
        text = _YEAH_FILLER_RE.sub(' ', text)
        text = _FILLER_REPLACE_SPACE.sub(' ', text)
        text = _REPEATED_WORD.sub(cls._dedupe_repeated_word, text)
        text = cls._normalize_spoken_acronyms(text)

        if dictionary:
            for wrong, right in sorted(dictionary.items(), key=lambda kv: -len(kv[0])):
                text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)

        text = cls._apply_self_corrections(text)
        text = cls._collapse_repeated_clauses(text)
        text = cls._dedupe_adjacent_sentences(text)
        text = cls._prune_low_information_fragments(text)
        if programmer_mode:
            text = cls._tag_file_mentions(text)
            text = cls._tag_symbol_mentions(text)
        text = cls._normalize_readability(text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r',([.!?])', r'\1', text)
        text = re.sub(r',\s*$', '', text)
        text = re.sub(r'^[,\s]+', '', text)
        return text.strip()

    @classmethod
    def clean_conservative(
        cls,
        text: str,
        dictionary: dict[str, str] | None = None,
        programmer_mode: bool = True,
    ) -> str:
        """Conservative cleanup that avoids sentence replacement heuristics."""
        for pattern in _FILLER_REMOVE:
            text = pattern.sub('', text)
        text = _LEADING_DISCOURSE.sub('', text)
        text = _INLINE_DISCOURSE_RE.sub(' ', text)
        text = _HESITATION_CHAIN_RE.sub('maybe', text)
        text = _YEAH_FILLER_RE.sub(' ', text)
        text = _FILLER_REPLACE_SPACE.sub(' ', text)
        text = _REPEATED_WORD.sub(cls._dedupe_repeated_word, text)
        text = cls._normalize_spoken_acronyms(text)
        if dictionary:
            for wrong, right in sorted(dictionary.items(), key=lambda kv: -len(kv[0])):
                text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
        text = cls._collapse_repeated_clauses(text)
        text = cls._dedupe_adjacent_sentences(text)
        text = cls._prune_low_information_fragments(text)
        if programmer_mode:
            text = cls._tag_file_mentions(text)
            text = cls._tag_symbol_mentions(text)
        text = cls._normalize_readability(text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r',([.!?])', r'\1', text)
        text = re.sub(r',\s*$', '', text)
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
                cue = cls._normalize_cue(inline.group("cue"))
                replacement = inline.group("replacement").strip(" ,.-")
                if cls._should_replace_previous(cue, prefix, replacement):
                    out.append(cls._merge_with_previous(prefix, replacement))
                else:
                    out.append(cls._ensure_terminal_punctuation(prefix))
                    out.append(cls._ensure_terminal_punctuation(replacement))
                continue
            match = _CORRECTION_PREFIX.match(sentence)
            if match:
                cue = cls._normalize_cue(match.group("cue"))
                replacement = sentence[match.end():].strip(" ,.-")
                if not replacement:
                    continue
                if out and cls._should_replace_previous(cue, out[-1], replacement):
                    out[-1] = cls._merge_with_previous(out[-1], replacement)
                else:
                    out.append(cls._ensure_terminal_punctuation(replacement))
                continue
            out.append(sentence)
        return " ".join(out)

    @staticmethod
    def _normalize_cue(cue: str) -> str:
        cue = cue.strip().lower().replace(",", " ")
        return re.sub(r"\s+", " ", cue)

    @classmethod
    def _should_replace_previous(cls, cue: str, previous: str, replacement: str) -> bool:
        if cue in _STRONG_REPLACE_CUES:
            return True
        if cue in _WEAK_REPLACE_CUES:
            # "sorry" appears in natural speech; only treat it as a correction
            # when the preceding fragment looks like a direct edit command.
            if (
                _VERB_TO_TARGET.match(previous)
                or _VERB_TRAILING_TOKEN.match(previous)
                or _ACTION_START_RE.match(previous)
            ) and len(replacement.split()) <= 10:
                return True
        return False

    @staticmethod
    def _ensure_terminal_punctuation(text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        if text.endswith((".", "!", "?")):
            return text
        return text + "."

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
        return cls._ensure_terminal_punctuation(replacement)

    @classmethod
    def _tag_file_mentions(cls, text: str) -> str:
        """Turn spoken or explicit file mentions into @-style file tags."""
        text = _SPOKEN_COMPLEX_FILE_RE.sub(cls._replace_spoken_complex_file, text)
        text = _SPOKEN_DOT_FILE_RE.sub(cls._replace_spoken_file, text)
        text = _EXPLICIT_FILE_RE.sub(cls._replace_explicit_file, text)
        text = _BARE_FILE_RE.sub(cls._replace_bare_file, text)
        text = _DUPLICATE_FILE_TAG_RE.sub("@", text)
        text = _LONE_EXTENSION_TAG_RE.sub(r"\g<ext>", text)
        text = _FRAGMENTED_TAG_RE.sub(cls._merge_fragmented_tags, text)
        text = _SPOKEN_FRAGMENTED_TAG_RE.sub(cls._merge_spoken_fragmented_tag, text)
        text = _VERB_PREFIX_TAG_FILE_RE.sub(cls._merge_prefixed_tagged_file, text)
        text = _TAGGED_JS_LIST_RE.sub(cls._untag_js_list, text)
        return text

    @classmethod
    def _tag_symbol_mentions(cls, text: str) -> str:
        """Tag explicit symbol mentions for programmer-focused prompts."""
        tagged = _SYMBOL_MENTION_RE.sub(cls._replace_symbol_mention, text)
        tagged = _DUPLICATE_SYMBOL_TAG_RE.sub(r"\1", tagged)
        return tagged

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
            if prev_norm and word_count >= 6 and prev_norm.endswith(norm):
                continue
            out.append(chunk)
            prev_norm = norm
        return " ".join(out) if out else text

    @staticmethod
    def _dedupe_adjacent_sentences(text: str) -> str:
        """Drop duplicated adjacent sentences while preserving order."""
        chunks = [chunk.strip() for chunk in _SENTENCE_SPLIT.split(text.strip()) if chunk.strip()]
        if len(chunks) < 2:
            return text

        out: list[str] = []
        prev_norm = ""
        for chunk in chunks:
            norm = TextCleaner._normalize_fragment(chunk)
            if norm and norm == prev_norm and len(norm.split()) >= 6:
                continue
            out.append(chunk)
            prev_norm = norm
        return " ".join(out) if out else text

    @staticmethod
    def _replace_spoken_file(match: re.Match[str]) -> str:
        base = match.group("base")
        ext = match.group("ext").lower()
        return f"@{base}.{ext}"

    @staticmethod
    def _replace_spoken_complex_file(match: re.Match[str]) -> str:
        base = match.group("base").strip()
        ext = match.group("ext").lower()
        base = re.sub(r"\s+(?:underscore|under score)\s+", "_", base, flags=re.IGNORECASE)
        base = re.sub(r"\s+(?:dash|hyphen)\s+", "-", base, flags=re.IGNORECASE)
        base = re.sub(r"\s+", "_", base)
        return f"@{base}.{ext}"

    @staticmethod
    def _replace_explicit_file(match: re.Match[str]) -> str:
        name = match.group("name")
        if name.lower() in _FRAMEWORK_FILE_TOKENS:
            return name
        return f"@{name}"

    @staticmethod
    def _replace_bare_file(match: re.Match[str]) -> str:
        base = match.group("base").strip()
        lowered = base.lower()
        if lowered in _GENERIC_FILE_BASES or lowered in _FILE_EXTS:
            return match.group(0)
        tag = re.sub(r"\s+", "_", base.strip())
        return f"@{tag}"

    @staticmethod
    def _replace_symbol_mention(match: re.Match[str]) -> str:
        full = match.group(0)
        name = match.group("name").strip()
        normalized = name.strip(".,!?;:")
        if not normalized:
            return full
        lowered = normalized.lower()
        if lowered in _GENERIC_SYMBOLS:
            return full
        if _SYMBOL_FILE_EXT_RE.search(normalized):
            return full
        if f"@{normalized}" in full:
            return full
        return f"{full} @{normalized}"

    @staticmethod
    def _merge_fragmented_tags(match: re.Match[str]) -> str:
        return f"@{match.group('left')}{match.group('sep')}{match.group('right')}"

    @staticmethod
    def _merge_spoken_fragmented_tag(match: re.Match[str]) -> str:
        sep = "_" if "under" in match.group("sep").lower() else "-"
        return f"@{match.group('left')}{sep}{match.group('right')}"

    @staticmethod
    def _merge_prefixed_tagged_file(match: re.Match[str]) -> str:
        verb = match.group("verb")
        middle = (match.group("middle") or "").strip()
        prefix = match.group("prefix")
        name = match.group("name")
        lowered_name = name.lower()
        lowered_prefix = prefix.lower()
        if lowered_name.startswith(f"{lowered_prefix}-") or lowered_name.startswith(
            f"{lowered_prefix}_"
        ):
            return match.group(0)
        if middle:
            return f"{verb} {middle} @{prefix}-{name}"
        return f"{verb} @{prefix}-{name}"

    @staticmethod
    def _untag_js_list(match: re.Match[str]) -> str:
        body = match.group("body").replace("@", "")
        return f"{match.group('prefix')}{body}"

    @classmethod
    def _normalize_spoken_acronyms(cls, text: str) -> str:
        text = _SPELLED_JS_RE.sub("JS", text)
        text = _SPELLED_TS_RE.sub("TS", text)

        def _js_homophone(match: re.Match[str]) -> str:
            base = match.group("base")
            if base.lower() in _JS_CONTEXT_HINTS:
                return f"{base} JS"
            return match.group(0)

        return _JS_HOMOPHONE_RE.sub(_js_homophone, text)

    @classmethod
    def _prune_low_information_fragments(cls, text: str) -> str:
        """Drop repetitive low-information discourse fragments from mixed sentences."""
        chunks = [chunk.strip() for chunk in _SOFT_CLAUSE_SPLIT_RE.split(text.strip()) if chunk.strip()]
        if len(chunks) < 2:
            return text

        normalized = [cls._normalize_fragment(chunk) for chunk in chunks]
        non_low_count = sum(0 if cls._is_low_info_fragment(norm) else 1 for norm in normalized)
        if non_low_count == 0:
            return chunks[0]

        out: list[str] = []
        previous_norm = ""
        for chunk, norm in zip(chunks, normalized):
            if not norm:
                continue
            if cls._is_low_info_fragment(norm):
                continue
            if norm == previous_norm:
                continue
            out.append(chunk)
            previous_norm = norm

        return " ".join(out) if out else chunks[0]

    @staticmethod
    def _normalize_fragment(text: str) -> str:
        stripped = _TRIM_EDGE_PUNCT_RE.sub("", text.lower())
        return re.sub(r"\s+", " ", stripped).strip()

    @staticmethod
    def _is_low_info_fragment(normalized: str) -> bool:
        return bool(_LOW_INFO_FRAGMENT_RE.match(normalized))

    @classmethod
    def _normalize_readability(cls, text: str) -> str:
        text = text.strip()
        text = text.rstrip(" ,;:")
        text = _TRAILING_CONJUNCTION_RE.sub("", text).rstrip(" ,;:")
        text = _MISSING_SENTENCE_BREAK_RE.sub(". ", text)
        text = _EMBEDDED_SHOULD_QUESTION_RE.sub(
            lambda m: f"if I ask, should {m.group('body').strip(' ,.;:')}? ",
            text,
        )
        text = _I_CONTRACTION_RE.sub("I", text)
        text = _STANDALONE_I_RE.sub("I", text)
        text = _LEADING_LOWER_RE.sub(lambda m: f"{m.group(1)}{m.group(2).upper()}", text)

        words = text.split()
        if len(words) >= 8 and not _TERMINAL_PUNCT_RE.search(text.strip()):
            text = text.rstrip() + "."
        return text
