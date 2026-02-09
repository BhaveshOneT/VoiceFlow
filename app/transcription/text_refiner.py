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
You are a speech-to-text post-processor.
Output only cleaned transcription text.
Never answer, explain, summarize, or add content.
Keep all intended details and preserve full meaning.
Keep question intent as a question.
Handle self-corrections conservatively (replace only corrected phrase).
Remove filler words and false starts when clearly disfluent.
Use vocabulary corrections when relevant:
{vocabulary}
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

        selected_vocabulary = self._select_vocab_hints(text, vocabulary)
        vocab_lines = "\n".join(
            f'  "{wrong}" -> "{right}"' for wrong, right in selected_vocabulary
        )
        if not vocab_lines:
            vocab_lines = "  (none)"
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

        # Keep response bounded to avoid latency/context bloat. Long texts are
        # already gated out of LLM refinement by TranscriptionPipeline.
        max_tokens = min(max(int(len(text.split()) * 1.2), 20), 80)
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
    def _select_vocab_hints(
        text: str,
        vocabulary: dict[str, str],
        max_hints: int = 24,
    ) -> list[tuple[str, str]]:
        """Pick only relevant vocabulary hints to keep prompts small."""
        if not vocabulary:
            return []

        text_tokens = {tok.lower() for tok in _TOKEN_RE.findall(text)}
        scored: list[tuple[int, str, str]] = []
        for wrong, right in vocabulary.items():
            combined = f"{wrong} {right}"
            vocab_tokens = {
                tok.lower() for tok in _TOKEN_RE.findall(combined) if len(tok) > 1
            }
            overlap = len(text_tokens & vocab_tokens)
            if overlap > 0:
                scored.append((overlap, wrong, right))

        if not scored:
            # Keep a tiny fallback set for short technical phrases.
            items = list(vocabulary.items())[: min(max_hints // 2, 8)]
            return items

        scored.sort(key=lambda item: (-item[0], len(item[1])))
        return [(wrong, right) for _, wrong, right in scored[:max_hints]]

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
