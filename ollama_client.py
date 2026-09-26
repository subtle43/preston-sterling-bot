from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from ollama import AsyncClient, ResponseError

from config import Settings
from memory import ChatMessage

log = logging.getLogger("ollama-discord")


class OllamaChat:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.ollama_model
        self.client = AsyncClient(host=settings.ollama_host, timeout=settings.request_timeout)
        # Models Ollama said cannot think; asked once, never again.
        self._no_think: set[str] = set()

    def num_ctx_for(self, model: str | None = None) -> int:
        """The context window to request for one specific model.

        The two models do not share a window. Sending one number for both is wrong
        in whichever direction the pair happens to sit: with a big default and a
        smaller heavy model the heavy call is asked for more than it accepts, and
        with a small local default and a large cloud heavy model the heavy model
        silently gets a fraction of the window it could have used.
        """
        heavy = self.settings.ollama_model_heavy
        if heavy and (model or self.model) == heavy:
            return self.settings.num_ctx_heavy
        return self.settings.num_ctx

    def _options(self, model: str | None = None) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": self.settings.temperature,
            "num_ctx": self.num_ctx_for(model),
        }
        if self.settings.num_predict is not None:
            options["num_predict"] = self.settings.num_predict
        return options

    async def close(self) -> None:
        await self.client.close()

    async def list_model_names(self) -> list[str]:
        listing = await self.client.list()
        names: list[str] = []
        for model in listing.models or []:
            name = getattr(model, "model", None) or getattr(model, "name", None)
            if name:
                names.append(str(name))
        return names

    async def ensure_model(self) -> None:
        names = await self.list_model_names()
        if self.model not in names:
            available = ", ".join(names) if names else "(none pulled)"
            raise RuntimeError(
                f"Ollama model {self.model!r} is not installed. "
                f"Pulled models: {available}. Run: ollama pull {self.model}"
            )

    async def warmup(self) -> None:
        try:
            await self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                think=False,
                keep_alive=self.settings.keep_alive,
                options={**self._options(), "num_predict": 1},
            )
            log.info("Warmed Ollama model %s", self.model)
        except Exception:
            log.exception("Could not warm Ollama model %s", self.model)

    def build_messages(
        self,
        system_prompt: str,
        history: list[ChatMessage],
        user_text: str,
        images: list[bytes] | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        user_message: dict[str, Any] = {"role": "user", "content": user_text}
        if images:
            # The Ollama client base64-encodes raw bytes for us.
            user_message["images"] = images
        messages.append(user_message)
        return messages

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        think: bool | str | None = None,
        num_predict: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        """Yield (content_delta, thinking_delta) chunks from Ollama.

        `model` overrides the default for one call, so a heavier model can take
        the questions that need it without changing the default for chatter.
        """
        use_think = self.settings.think if think is None else think
        options = self._options(model)
        if num_predict is not None:
            options["num_predict"] = num_predict
        if temperature is not None:
            options["temperature"] = temperature
        target = model or self.model
        if use_think and target in self._no_think:
            use_think = False
        yielded = False
        try:
            stream = await self.client.chat(
                model=target, messages=messages, stream=True, think=use_think,
                keep_alive=self.settings.keep_alive, options=options,
            )
            async for part in stream:
                message = part.message
                content = message.content or ""
                thinking = getattr(message, "thinking", None) or ""
                if content or thinking:
                    yielded = True
                    yield content, thinking
            return
        except ResponseError as exc:
            # "hauhau-e4b:q4 does not support thinking": the summary fold asks for
            # think="low" (a Gemini level) and failed on every local model, so the
            # channel memory never folded and grew to its cap. Thinking is a
            # preference - drop it, remember the model, and ask again. The error
            # arrives on the first chunk, before anything was yielded.
            if yielded or not use_think or "does not support thinking" not in str(exc):
                raise
            self._no_think.add(target)
            log.info("%s does not support thinking - retrying without it", target)
        stream = await self.client.chat(
            model=target, messages=messages, stream=True, think=False,
            keep_alive=self.settings.keep_alive, options=options,
        )
        async for part in stream:
            message = part.message
            content = message.content or ""
            thinking = getattr(message, "thinking", None) or ""
            if content or thinking:
                yield content, thinking
