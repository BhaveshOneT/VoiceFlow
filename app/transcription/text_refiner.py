from __future__ import annotations

import gc
import logging
import re

log = logging.getLogger(__name__)

DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
_QUESTION_START_RE = re.compile(
    r"^\s*(who|what|when|where|why|how|is|are|am|was|were|do|does|did|can|"
    r"could|should|would|will|which|whose|whom|what's|whats|isn't|aren't|"
    r"won't|can't|couldn't|shouldn't|wouldn't|wer|was|wann|wo|warum|wie|"
    r"ist|sind|bin|war|waren|kann|kannst|können|soll|sollte|würde|"
    r"hat|haben|gibt|gibt's)\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")
_COMMON_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "this",
    "to",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}
_ANSWER_START_RE = re.compile(
    r"^\s*(yes|no|it\s+is|it's|this\s+is|the\s+answer|you\s+can|you\s+should|"
    r"because|in\s+summary|to\s+answer|ja|nein|die\s+antwort|"
    r"du\s+kannst|sie\s+können|weil|kurz\s+gesagt)\b",
    re.IGNORECASE,
)
_ASSISTANTY_START_RE = re.compile(
    r"^\s*(sure|certainly|absolutely|here(?:'s| is)|let's|i can|"
    r"you can|to do this|first,|here are|this version)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT_TEMPLATE = """\
You are a speech-to-text post-processor for a software developer.
Your task is to correct transcription errors in dictated text.

VOCABULARY CORRECTIONS — fix these commonly misheard terms:
{vocabulary}

RULES:
1. Fix misheard technical terms using the vocabulary list above
2. Remove filler words (um, uh, like, you know, basically)
3. Handle self-corrections conservatively: replace only the corrected phrase \
and keep all unrelated context and sentences. \
Backtracking signals include: "sorry", "I mean", "I meant", "actually", \
"no wait", "wait no", "scratch that", "never mind that", "not that", \
"let me rephrase", "correction", "rather".
4. Remove false starts only when they are clearly incomplete fragments.
5. Preserve camelCase, PascalCase, snake_case as appropriate for code terms
6. Fix grammar and add natural punctuation
7. Do NOT add information that was not spoken
8. Do NOT summarize — keep all intended content
9. Do NOT answer questions in the text
10. Do NOT provide explanations, suggestions, or completions
11. Never drop complete trailing clauses or end mid-thought
12. Output ONLY the cleaned transcript text, nothing else

SELF-CORRECTION EXAMPLES:
- "change it to red, sorry, blue" -> "change it to blue"
- "call the function foo, actually bar" -> "call the function bar"
- "set the font size to 12, no wait, 14 pixels" -> "set the font size to 14 pixels"
- "import react, I mean import react dom" -> "import react dom"
- "delete the file, scratch that, just rename it" -> "just rename it"
- "use a for loop, never mind that, use map instead" -> "use map instead"
- "send it to the staging, wait no, production server" -> "send it to the production server"
- "I want to use, no wait, I mean useState" -> "I want to use useState"

QUESTION EXAMPLES:
- "what is polymorphism in oop" -> "What is polymorphism in OOP?"
- "how do I reset my api key" -> "How do I reset my API key?"
"""


class TextRefiner:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.loaded:
            return
        from mlx_lm import load  # type: ignore[import-untyped]

        log.info("Loading LLM %s", self.model_name)
        self.model, self.tokenizer = load(self.model_name)
        log.info("LLM loaded")

    def unload(self) -> None:
        if not self.loaded:
            return
        import mlx.core as mx  # type: ignore[import-untyped]

        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        mx.clear_cache()
        log.info("LLM unloaded")

    def refine(self, text: str, vocabulary: dict[str, str]) -> str:
        if not self.loaded:
            self.load()

        vocab_lines = "\n".join(
            f'  "{wrong}" -> "{right}"' for wrong, right in vocabulary.items()
        )
        system = SYSTEM_PROMPT_TEMPLATE.format(vocabulary=vocab_lines)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        from mlx_lm import generate  # type: ignore[import-untyped]
        from mlx_lm.sample_utils import make_sampler  # type: ignore[import-untyped]

        # Keep response bounded, but leave enough room to avoid truncating
        # medium-length dictation that still needs punctuation cleanup.
        max_tokens = min(max(int(len(text.split()) * 1.8), 32), 128)
        sampler = make_sampler(temp=0.0)
        result = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )
        candidate = self._sanitize_output(result)
        if not candidate:
            return ""
        if self._is_answer_like(source=text, candidate=candidate):
            log.warning(
                "Rejected LLM refinement that changed intent or looked like an answer."
            )
            return ""
        return candidate

    @staticmethod
    def _sanitize_output(result: str) -> str:
        """Strip prompt leakage / meta responses from model output."""
        text = result.strip()
        if not text:
            return ""

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines or [text]:
            candidate = line.strip().strip("`").strip()
            candidate = re.sub(r"^\s*[-*]\s+", "", candidate)
            candidate = re.sub(
                r'^(cleaned text|corrected text|revised text|output|answer|response|'
                r'explanation|final|result)\s*:\s*',
                '',
                candidate,
                flags=re.IGNORECASE,
            ).strip()
            candidate = candidate.strip('"').strip("'").strip()
            if not candidate:
                continue
            lower = candidate.lower()
            leak_markers = (
                "you are a",
                "system prompt",
                "rules:",
                "self-correction examples",
                "as an ai",
                "this version is concise",
                "this version is",
                "directly addresses the question",
                "refined version",
                "rewritten version",
                "concise and directly",
            )
            if any(marker in lower for marker in leak_markers):
                continue
            return candidate
        return ""

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        stripped = text.strip()
        return stripped.endswith("?") or bool(_QUESTION_START_RE.match(stripped))

    @staticmethod
    def _keywords(text: str) -> set[str]:
        tokens = [tok.lower() for tok in _TOKEN_RE.findall(text)]
        return {
            token
            for token in tokens
            if len(token) > 2 and token not in _COMMON_WORDS
        }

    @classmethod
    def _is_answer_like(cls, source: str, candidate: str) -> bool:
        """Detect when model output drifts into generated answers."""
        source_words = source.split()
        candidate_words = candidate.split()
        if len(candidate_words) > max((len(source_words) * 2), len(source_words) + 12):
            return True

        lower_candidate = candidate.strip().lower()
        if lower_candidate.startswith(("answer:", "response:", "explanation:")):
            return True
        if _ASSISTANTY_START_RE.match(candidate) and not _ASSISTANTY_START_RE.match(source):
            return True

        source_is_question = cls._looks_like_question(source)
        candidate_is_question = cls._looks_like_question(candidate)
        if source_is_question:
            if _ANSWER_START_RE.match(lower_candidate):
                return True
            # Preserve question intent; avoid converting spoken questions into answers.
            if not candidate_is_question and not _QUESTION_START_RE.match(lower_candidate):
                return True

        source_keywords = cls._keywords(source)
        candidate_keywords = cls._keywords(candidate)
        if candidate_keywords:
            new_tokens = candidate_keywords - source_keywords
            novelty_ratio = len(new_tokens) / len(candidate_keywords)
            if novelty_ratio > 0.45 and len(candidate_keywords) >= 6:
                return True
        return False
