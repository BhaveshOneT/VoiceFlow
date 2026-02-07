from __future__ import annotations

import gc
import logging

log = logging.getLogger(__name__)

DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

SYSTEM_PROMPT_TEMPLATE = """\
You are a speech-to-text post-processor for a software developer.
Your task is to correct transcription errors in dictated text.

VOCABULARY CORRECTIONS — fix these commonly misheard terms:
{vocabulary}

RULES:
1. Fix misheard technical terms using the vocabulary list above
2. Remove filler words (um, uh, like, you know, basically)
3. Handle self-corrections: when the speaker backtracks, DISCARD everything \
before the correction signal and keep ONLY the corrected version. \
Backtracking signals include: "sorry", "I mean", "I meant", "actually", \
"no wait", "wait no", "scratch that", "never mind that", "not that", \
"let me rephrase", "correction", "rather".
4. Remove false starts. Keep only the final intended version.
5. Preserve camelCase, PascalCase, snake_case as appropriate for code terms
6. Fix grammar and add natural punctuation
7. Do NOT add information that was not spoken
8. Do NOT summarize — keep all intended content
9. Output ONLY the cleaned text, nothing else

SELF-CORRECTION EXAMPLES:
- "change it to red, sorry, blue" -> "change it to blue"
- "call the function foo, actually bar" -> "call the function bar"
- "set the font size to 12, no wait, 14 pixels" -> "set the font size to 14 pixels"
- "import react, I mean import react dom" -> "import react dom"
- "delete the file, scratch that, just rename it" -> "just rename it"
- "use a for loop, never mind that, use map instead" -> "use map instead"
- "send it to the staging, wait no, production server" -> "send it to the production server"
- "I want to use, no wait, I mean useState" -> "I want to use useState\""""


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

        max_tokens = min(max(int(len(text.split()) * 3), 64), 512)
        sampler = make_sampler(temp=0.1)
        result = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )
        return result.strip()
