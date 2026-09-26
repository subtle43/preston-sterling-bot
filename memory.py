from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Deque, TypedDict

log = logging.getLogger("ollama-discord")

# Keep the persisted file bounded no matter how busy the server gets.
MAX_PERSISTED_CHANNELS = 200


class ChatMessage(TypedDict):
    role: str
    content: str


class ConversationStore:
    """Chat history keyed by Discord channel (or DM user).

    Lives in RAM; optionally backed by a JsonStore so it survives a restart.

    Two layers per key: the recent messages VERBATIM, and a rolling SUMMARY of
    everything that has scrolled out of the verbatim window. Messages that fall
    off the window are not thrown away - they queue in `pending` until the bot
    folds them into the summary (bot.py does the folding; it needs the model).
    The model is stateless, so this store is the bot's entire memory of a
    channel, and the summary is how a channel's history outlives the window.
    """

    def __init__(self, max_messages: int, char_budget: int, store=None) -> None:
        self.max_messages = max_messages
        self.char_budget = char_budget
        self.store = store
        self._histories: dict[str, Deque[ChatMessage]] = defaultdict(deque)
        self._summaries: dict[str, str] = {}
        self._pending: dict[str, list[ChatMessage]] = defaultdict(list)

    def load(self) -> int:
        """Restore histories from disk. Bad entries are skipped, never fatal."""
        if self.store is None:
            return 0
        raw = self.store.load()
        restored = 0
        # Old format: {key: [messages]}. New: {"histories": {...}, "summaries":
        # {...}, "pending": {...}}. Accept both, forever.
        if isinstance(raw.get("histories"), dict):
            for key, text in (raw.get("summaries") or {}).items():
                if isinstance(key, str) and isinstance(text, str) and text.strip():
                    self._summaries[key] = text
            for key, items in (raw.get("pending") or {}).items():
                if isinstance(key, str) and isinstance(items, list):
                    self._pending[key] = [
                        {"role": i["role"], "content": i["content"]}
                        for i in items
                        if isinstance(i, dict) and isinstance(i.get("role"), str)
                        and isinstance(i.get("content"), str)
                    ]
            raw = raw["histories"]
        for key, items in raw.items():
            if not isinstance(key, str) or not isinstance(items, list):
                continue
            history: Deque[ChatMessage] = deque()
            for item in items:
                if (
                    isinstance(item, dict)
                    and isinstance(item.get("role"), str)
                    and isinstance(item.get("content"), str)
                ):
                    history.append({"role": item["role"], "content": item["content"]})
            # A question with no answer after it is a save that landed between
            # the two halves of an exchange (the user turn is written, the reply
            # is still in the debounce window) and then an unclean exit. Left
            # in, the model treats it as an open question and answers THAT the
            # next time somebody says "sup bro". Drop it; the reply was posted.
            while history and history[-1].get("role") == "user":
                history.pop()
            if history:
                self._trim(history, key)
                self._histories[key] = history
                restored += 1
        if restored:
            log.info(
                "Restored conversation history for %d channels (%d with summaries)",
                restored, sum(1 for k in self._summaries if k in self._histories),
            )
        return restored

    def _snapshot(self) -> dict:
        items = list(self._histories.items())[-MAX_PERSISTED_CHANNELS:]
        keys = {k for k, v in items if v}
        return {
            "histories": {k: list(v) for k, v in items if v},
            "summaries": {k: t for k, t in self._summaries.items() if k in keys or t},
            "pending": {k: v for k, v in self._pending.items() if v},
        }

    def _persist(self) -> None:
        if self.store is None:
            return
        # Most recently touched channels win if we are over the cap.
        self.store.data = self._snapshot()
        self.store.touch()
        self.store.maybe_save()

    def flush_if_dirty(self) -> None:
        """Write anything the debounce is still holding. Cheap when clean."""
        if self.store is not None and getattr(self.store, "_dirty", False):
            self.store.save()

    def flush(self) -> None:
        """Force a save, e.g. on shutdown."""
        if self.store is None:
            return
        self.store.data = self._snapshot()
        self.store.touch()
        self.store.save()

    def key_for(self, guild_id: int | None, channel_id: int, is_dm: bool, user_id: int) -> str:
        if is_dm:
            return f"dm:{user_id}"
        return f"guild:{guild_id or 0}:channel:{channel_id}"

    def get(self, key: str) -> list[ChatMessage]:
        return list(self._histories[key])

    def add(self, key: str, role: str, content: str) -> None:
        history = self._histories[key]
        history.append({"role": role, "content": content})
        self._trim(history, key)
        self._persist()

    def clear(self, key: str) -> int:
        history = self._histories.get(key)
        count = len(history) if history else 0
        self._histories[key] = deque()
        self._summaries.pop(key, None)
        self._pending.pop(key, None)
        self._persist()
        return count

    # -- the summary layer --------------------------------------------------

    def summary(self, key: str) -> str:
        return self._summaries.get(key, "")

    def set_summary(self, key: str, text: str) -> None:
        text = (text or "").strip()
        if text:
            self._summaries[key] = text
        else:
            self._summaries.pop(key, None)
        self._persist()

    def pending(self, key: str) -> list[ChatMessage]:
        return list(self._pending.get(key, []))

    def pending_chars(self, key: str) -> int:
        return sum(len(m["content"]) for m in self._pending.get(key, []))

    def take_pending(self, key: str) -> list[ChatMessage]:
        """Hand over the queued messages and empty the queue."""
        items = self._pending.pop(key, [])
        self._persist()
        return items

    def push_out(self, key: str, keep_last: int) -> int:
        """Move everything but the newest `keep_last` messages into the pending
        queue, so a fold can summarise it now. Returns how many moved."""
        history = self._histories.get(key)
        if not history or len(history) <= keep_last:
            return 0
        moved: list[ChatMessage] = []
        while len(history) > keep_last:
            moved.append(history.popleft())
        self._pending[key].extend(moved)
        self._persist()
        return len(moved)

    def stats(self, key: str) -> dict:
        history = self._histories.get(key) or []
        return {
            "messages": len(history),
            "chars": sum(len(m["content"]) for m in history),
            "pending": len(self._pending.get(key, [])),
            "summary_words": len(self._summaries.get(key, "").split()),
        }

    def requeue(self, key: str, items: list[ChatMessage]) -> None:
        """Put messages back at the FRONT of the queue after a failed fold."""
        self._pending[key] = items + self._pending.get(key, [])
        self._persist()

    def _trim(self, history: Deque[ChatMessage], key: str | None = None) -> None:
        dropped: list[ChatMessage] = []
        while len(history) > self.max_messages:
            dropped.append(history.popleft())
        while history and sum(len(item["content"]) for item in history) > self.char_budget:
            dropped.append(history.popleft())
        if dropped and key is not None:
            queue = self._pending[key]
            queue.extend(dropped)
            # A queue that nobody folds must not grow forever (e.g. summaries
            # disabled): keep the newest slice, at most one window's worth.
            if len(queue) > self.max_messages:
                del queue[: len(queue) - self.max_messages]
