"""Gemini backend, shaped exactly like OllamaChat so bot.py cannot tell them apart.

Same method names, same arguments, same (content, thinking) stream. The whole
point is that the call sites in bot.py stay untouched - there is one place that
chooses a backend and nothing else in the program knows which one it got.

Talks to the REST API over aiohttp rather than pulling in google-genai: aiohttp is
already a dependency for the web search, and the surface used here is two
endpoints. Nothing about a model this far away needs a whole SDK.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import aiohttp

from config import Settings
from memory import ChatMessage

log = logging.getLogger("ollama-discord")

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"

# The API's adjustable content filters, all at their most permissive: the persona
# is deliberately vulgar, and a blocked reply is an empty one. The model's own
# training (and Google's non-adjustable filters) still apply on top.
SAFETY_SETTINGS = [
    {"category": c, "threshold": "BLOCK_NONE"}
    for c in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
              "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
]

# What the API accepts for thinkingConfig.thinkingLevel.
THINKING_LEVELS = {"MINIMAL", "LOW", "MEDIUM", "HIGH"}

# Extra output tokens granted so the reasoning does not eat the visible answer.
THINKING_HEADROOM = {"MINIMAL": 0, "LOW": 2000, "MEDIUM": 4000, "HIGH": 8000}


class _ThinkingLevelRejected(Exception):
    """The model refused the requested thinkingLevel. Internal, never surfaced."""


class GeminiBusy(RuntimeError):
    """A transient refusal (429/500/502/503/504) - "model is experiencing high
    demand". Retried in stream_chat before it reaches a caller."""


RETRYABLE = {429, 500, 502, 503, 504}
RETRY_DELAYS = (2.0, 5.0)


class GeminiChat:
    # stream_chat accepts response_schema=: the reply is constrained to JSON of
    # that shape. OllamaChat has no such thing; callers check this attribute.
    supports_json_schema = True

    """Drop-in replacement for OllamaChat, backed by the Gemini API."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.gemini_model
        self._session: aiohttp.ClientSession | None = None

    # -- plumbing ---------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.settings.request_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    def num_ctx_for(self, model: str | None = None) -> int:
        """The window log budgets should be sized against.

        Not a request parameter - Gemini manages its own window - but bot.py has to
        decide how much of a CSV it may paste in, and that answer is completely
        different here. Sized against OLLAMA_NUM_CTX the bot was trimming logs to
        about 6 KB while the API would accept a megabyte.
        """
        return self.settings.gemini_num_ctx

    async def list_model_names(self) -> list[str]:
        session = await self._get_session()
        url = f"{ENDPOINT}/models?key={self.settings.gemini_api_key}"
        async with session.get(url) as resp:
            body = await resp.json()
        return [
            str(m.get("name", "")).replace("models/", "")
            for m in (body.get("models") or [])
        ]

    async def ensure_model(self) -> None:
        """Fail at startup rather than on the first question."""
        if not self.settings.gemini_api_key:
            raise RuntimeError("GEMINI_MODEL is set but GEMINI_API_KEY is empty.")
        try:
            names = await self.list_model_names()
        except Exception as exc:
            raise RuntimeError(f"Could not reach the Gemini API: {exc}") from exc
        if names and self.model not in names:
            near = [n for n in names if n.startswith(self.model.split("-")[0])][:8]
            raise RuntimeError(
                f"Gemini model {self.model!r} is not available to this key. "
                f"Close matches: {', '.join(near) or '(none)'}"
            )

    async def warmup(self) -> None:
        """Nothing to load, but prove the key works before anyone asks a question."""
        try:
            await self.ensure_model()
            log.info("Gemini reachable, model %s", self.model)
        except Exception:
            log.exception("Could not reach Gemini model %s", self.model)

    # -- request shaping --------------------------------------------------

    def build_messages(
        self,
        system_prompt: str,
        history: list[ChatMessage],
        user_text: str,
        images: list[bytes] | None = None,
    ) -> list[dict[str, Any]]:
        """Identical output shape to OllamaChat.build_messages.

        Kept in the Ollama vocabulary (system/user/assistant, raw image bytes) and
        translated at the last moment in _payload. bot.py builds these lists in
        several places and must not have to care which backend is live.
        """
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        user_message: dict[str, Any] = {"role": "user", "content": user_text}
        if images:
            user_message["images"] = images
        messages.append(user_message)
        return messages

    @staticmethod
    def _parts(message: dict[str, Any]) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        text = str(message.get("content") or "")
        if text:
            parts.append({"text": text})
        for blob in message.get("images") or []:
            if not isinstance(blob, (bytes, bytearray)):
                continue
            parts.append({
                "inlineData": {
                    "mimeType": _sniff_mime(bytes(blob)),
                    "data": base64.b64encode(bytes(blob)).decode("ascii"),
                }
            })
        return parts

    def _payload(
        self,
        messages: list[dict[str, Any]],
        *,
        think: bool | str | None,
        num_predict: int | None,
        temperature: float | None,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        system_bits: list[str] = []
        contents: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                # Gemini takes the system prompt out of band. Several are collapsed
                # into one because bot.py appends extra blocks as separate turns.
                text = str(message.get("content") or "")
                if text:
                    system_bits.append(text)
                continue
            parts = self._parts(message)
            if not parts:
                continue
            contents.append(
                {"role": "model" if role == "assistant" else "user", "parts": parts}
            )

        generation: dict[str, Any] = {
            "temperature": (
                self.settings.temperature if temperature is None else temperature
            ),
        }
        budget = num_predict if num_predict is not None else self.settings.num_predict
        level = _thinking_level(think, self.settings.gemini_thinking_level)
        if level:
            generation["thinkingConfig"] = {"thinkingLevel": level}
        if budget:
            # Reasoning is spent out of the SAME maxOutputTokens pot as the answer,
            # so a raised thinking level on an unchanged budget buys silence: the
            # model thinks its way through the allowance and streams nothing. This
            # is the identical trap reply_tokens() already documents for Ollama,
            # which adds 2500 when thinking is on - the headroom just has to be
            # added here too, because bot.py sizes for the local backend.
            generation["maxOutputTokens"] = int(budget) + THINKING_HEADROOM.get(level, 0)
        if response_schema:
            # Structured output: no fences, no prose around the object, and the
            # enums are enforced by the API rather than checked afterwards.
            generation["responseMimeType"] = "application/json"
            generation["responseSchema"] = response_schema

        payload: dict[str, Any] = {"contents": contents, "generationConfig": generation,
                                   "safetySettings": SAFETY_SETTINGS}
        if system_bits:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_bits)}]
            }
        return payload

    # -- streaming --------------------------------------------------------

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        think: bool | str | None = None,
        num_predict: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        """Yield (content_delta, thinking_delta), same contract as OllamaChat.

        Gemini marks reasoning parts with "thought": true. Those go down the
        thinking channel so SHOW_THINKING keeps working and, more importantly, so
        reasoning never lands in the visible reply by accident.
        """
        payload = self._payload(
            messages, think=think, num_predict=num_predict, temperature=temperature,
            response_schema=response_schema,
        )
        target = model or self.model
        # "High demand" 503s come in bursts and are model-specific: a topic song
        # died on one with nothing retried. Retry while nothing has been yielded
        # (so no text is ever duplicated), the last attempt on the fallback model.
        fallback = getattr(self.settings, "gemini_fallback_model", "") or ""
        attempts = [target, target] + ([fallback] if fallback and fallback != target else [target])
        for n, current in enumerate(attempts):
            url = (
                f"{ENDPOINT}/models/{current}:streamGenerateContent"
                f"?alt=sse&key={self.settings.gemini_api_key}"
            )
            yielded = False
            try:
                async for pair in self._stream_with_level_retry(url, payload, current):
                    yielded = True
                    yield pair
                return
            except GeminiBusy as exc:
                if yielded or n == len(attempts) - 1:
                    raise
                delay = RETRY_DELAYS[min(n, len(RETRY_DELAYS) - 1)]
                log.warning("%s busy (%s) - retrying in %.0fs%s", current, str(exc)[:40], delay,
                            f" on {attempts[n + 1]}" if attempts[n + 1] != current else "")
                await asyncio.sleep(delay)

    async def _stream_with_level_retry(
        self, url: str, payload: dict[str, Any], target: str
    ) -> AsyncIterator[tuple[str, str]]:
        try:
            async for pair in self._stream_once(url, payload, target):
                yield pair
        except _ThinkingLevelRejected:
            # Not every model accepts every level - gemini-3.8-flash rejects
            # MINIMAL with a hard 400. The level is a preference, never the point
            # of the question, so drop it and ask again rather than losing the
            # reply over it. Raised before any content is yielded, so the retry
            # cannot duplicate text already sent to the channel.
            log.info("%s rejected thinkingLevel - retrying without it", target)
            payload["generationConfig"].pop("thinkingConfig", None)
            async for pair in self._stream_once(url, payload, target):
                yield pair

    async def _stream_once(
        self, url: str, payload: dict[str, Any], target: str
    ) -> AsyncIterator[tuple[str, str]]:
        session = await self._get_session()
        async with session.post(
            url, json=payload, headers={"Content-Type": "application/json"}
        ) as resp:
            if resp.status != 200:
                detail = (await resp.text())[:400]
                if (
                    resp.status == 400
                    and "hinking level" in detail
                    and "thinkingConfig" in payload.get("generationConfig", {})
                ):
                    raise _ThinkingLevelRejected(detail)
                if resp.status in RETRYABLE:
                    raise GeminiBusy(f"Gemini HTTP {resp.status}: {detail}")
                raise RuntimeError(f"Gemini HTTP {resp.status}: {detail}")
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if not body or body == "[DONE]":
                    continue
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                for candidate in chunk.get("candidates") or []:
                    for part in (candidate.get("content") or {}).get("parts") or []:
                        text = part.get("text") or ""
                        if not text:
                            continue
                        if part.get("thought"):
                            yield "", text
                        else:
                            yield text, ""


def _thinking_level(think: bool | str | None, configured: str) -> str | None:
    """Map the Ollama-style `think` argument onto a Gemini thinking level.

    bot.py turns thinking on and off per call - a log review deliberates, a
    one-liner does not - and that decision has to survive the backend swap. False
    is MINIMAL rather than nothing at all: the API has no true "off" and omitting
    the config lets the model spend as much as it likes.
    """
    # An explicit GEMINI_THINKING_LEVEL is an operator decision and outranks the
    # per-call flag. It has to: bot.py passes think=False on ordinary chat and
    # OLLAMA_THINK is false, so the old order pinned everything to MINIMAL and the
    # setting could never do anything at all.
    # ...except for a call that names its own level. A judge call - the intent
    # classifier, the yes/no routers - passes think="minimal" because a hundred
    # tokens of JSON do not need MEDIUM deliberation, and with the operator's
    # level applied to it the classifier was taking one to four seconds.
    if isinstance(think, str) and think.upper() in THINKING_LEVELS:
        return think.upper()
    level = (configured or "").strip().upper()
    if level in THINKING_LEVELS:
        return level
    if think is False:
        return "MINIMAL"
    if isinstance(think, str) and think.upper() in THINKING_LEVELS:
        return think.upper()
    if isinstance(think, str):
        mapped = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}.get(think.lower())
        if mapped:
            return mapped
    return None


def _sniff_mime(blob: bytes) -> str:
    """Discord hands over bytes with no content type, and Gemini insists on one."""
    if blob.startswith(b"\x89PNG"):
        return "image/png"
    if blob.startswith(b"GIF8"):
        return "image/gif"
    if blob.startswith(b"RIFF") and blob[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"
