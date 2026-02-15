"""LLM-based meeting summarization.

Produces structured output: summary, key points, and action items.
Uses Qwen2.5-0.5B locally by default, with optional OpenAI API backend
for higher quality.

For long transcripts (>2000 words), uses a map-reduce strategy:
chunk the transcript, summarize each chunk, then meta-summarize.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from app.config import DEFAULT_LLM_MODEL

log = logging.getLogger(__name__)

_WORD_LIMIT_SINGLE_PASS = 2000
_CHUNK_SIZE_WORDS = 1500
_CHUNK_OVERLAP_WORDS = 100

_SYSTEM_PROMPT = """\
You are a meeting summarizer. Given a meeting transcript, produce a structured summary.
Output valid JSON with exactly these fields:
{
  "summary": "2-4 sentence summary of the meeting",
  "key_points": ["point 1", "point 2", ...],
  "action_items": ["action 1", "action 2", ...]
}
Only output the JSON. No preamble, no explanation.
If there are no clear action items, return an empty list.
"""

_META_SYSTEM_PROMPT = """\
You are a meeting summarizer. Given multiple partial summaries of a long meeting, \
produce a single cohesive summary.
Output valid JSON with exactly these fields:
{
  "summary": "2-4 sentence summary of the entire meeting",
  "key_points": ["point 1", "point 2", ...],
  "action_items": ["action 1", "action 2", ...]
}
Only output the JSON. Combine and deduplicate the key points and action items.
"""


@dataclass
class SummaryResult:
    summary_text: str
    key_points: list[str]
    action_items: list[str]


class MeetingSummarizer:
    """Generates meeting summaries from transcript text.

    Supports two backends:
    - ``local``: Uses Qwen2.5-0.5B via mlx-lm (default)
    - ``openai``: Uses OpenAI API (requires api_key)
    """

    def __init__(
        self,
        provider: str = "local",
        model: str = DEFAULT_LLM_MODEL,
        api_key: str = "",
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key

    def summarize(self, transcript_text: str) -> SummaryResult:
        """Summarize transcript text and return structured result."""
        words = transcript_text.split()
        word_count = len(words)
        log.info("Summarizing transcript (%d words, provider=%s)", word_count, self._provider)

        if word_count <= _WORD_LIMIT_SINGLE_PASS:
            return self._single_pass(transcript_text)
        return self._map_reduce(transcript_text, words)

    def _single_pass(self, text: str) -> SummaryResult:
        """Summarize a short transcript in one LLM call."""
        prompt = f"Meeting transcript:\n\n{text}"
        response = self._call_llm(_SYSTEM_PROMPT, prompt)
        return self._parse_response(response)

    def _map_reduce(self, text: str, words: list[str]) -> SummaryResult:
        """Chunk, summarize each chunk, then meta-summarize."""
        # Split into overlapping chunks
        chunks: list[str] = []
        i = 0
        while i < len(words):
            end = min(i + _CHUNK_SIZE_WORDS, len(words))
            chunk_text = " ".join(words[i:end])
            chunks.append(chunk_text)
            i += _CHUNK_SIZE_WORDS - _CHUNK_OVERLAP_WORDS

        log.info("Map-reduce: %d chunks from %d words", len(chunks), len(words))

        # Map: summarize each chunk
        chunk_summaries: list[str] = []
        for idx, chunk in enumerate(chunks):
            prompt = f"Meeting transcript (part {idx + 1} of {len(chunks)}):\n\n{chunk}"
            try:
                response = self._call_llm(_SYSTEM_PROMPT, prompt)
                chunk_summaries.append(response)
            except Exception:
                log.warning("Chunk %d summarization failed, skipping", idx + 1)
                continue

        if not chunk_summaries:
            return SummaryResult(
                summary_text="Summarization failed for all chunks.",
                key_points=[],
                action_items=[],
            )

        # Reduce: meta-summarize
        combined = "\n\n---\n\n".join(
            f"Partial summary {i + 1}:\n{s}" for i, s in enumerate(chunk_summaries)
        )
        meta_response = self._call_llm(_META_SYSTEM_PROMPT, combined)
        return self._parse_response(meta_response)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        if self._provider == "openai":
            return self._call_openai(system_prompt, user_prompt)
        return self._call_local(system_prompt, user_prompt)

    def _call_local(self, system_prompt: str, user_prompt: str) -> str:
        """Call Qwen via mlx-lm."""
        from mlx_lm import generate, load  # type: ignore[import-untyped]

        model, tokenizer = load(self._model)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response = generate(
            model,
            tokenizer,
            prompt=prompt_text,
            max_tokens=1024,
            temp=0.3,
        )
        return response.strip()

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI API."""
        import urllib.request

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        body = json.dumps({
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
        }).encode()

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()

    @staticmethod
    def _parse_response(response: str) -> SummaryResult:
        """Parse JSON from LLM response, with fallback for malformed output."""
        # Try to extract JSON from the response
        # LLMs sometimes wrap JSON in markdown code blocks
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return SummaryResult(
                    summary_text=data.get("summary", ""),
                    key_points=data.get("key_points", []),
                    action_items=data.get("action_items", []),
                )
            except json.JSONDecodeError:
                pass

        # Fallback: treat entire response as summary text
        log.warning("Could not parse structured JSON from LLM response")
        return SummaryResult(
            summary_text=response[:500],
            key_points=[],
            action_items=[],
        )
