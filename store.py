from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("ollama-discord")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# Hard ceiling so a busy server cannot fill the disk.
MAX_STATE_BYTES = 8 * 1024 * 1024


class JsonStore:
    """A JSON file on disk, written atomically, saved at most every few seconds.

    Deliberately inert: json only, never pickle, never eval. A corrupt or
    hand-edited file is treated as untrusted and falls back to empty rather than
    crashing the bot on startup.

    The filename is fixed by the caller - never build it from user input.
    """

    def __init__(
        self,
        filename: str,
        *,
        save_interval: float = 5.0,
        max_bytes: int = MAX_STATE_BYTES,
    ) -> None:
        self.path = DATA_DIR / filename
        self.save_interval = save_interval
        self.max_bytes = max_bytes
        self.data: dict[str, Any] = {}
        self._dirty = False
        self._last_save = 0.0

    def load(self) -> dict[str, Any]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self.data = {}
            return self.data
        except OSError:
            log.exception("Could not read %s, starting empty", self.path.name)
            self.data = {}
            return self.data
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            log.warning("%s is not valid JSON, starting empty", self.path.name)
            self.data = {}
            return self.data
        self.data = parsed if isinstance(parsed, dict) else {}
        log.info("Loaded %s (%d entries)", self.path.name, len(self.data))
        return self.data

    def touch(self) -> None:
        self._dirty = True

    def maybe_save(self) -> None:
        """Save if dirty and the debounce window has passed."""
        if not self._dirty:
            return
        if time.monotonic() - self._last_save < self.save_interval:
            return
        self.save()

    def save(self) -> None:
        try:
            payload = json.dumps(self.data, ensure_ascii=False)
        except (TypeError, ValueError):
            log.exception("Could not serialise %s", self.path.name)
            return
        size = len(payload.encode("utf-8"))
        if size > self.max_bytes:
            log.warning(
                "%s is %d bytes, over the %d cap - callers should prune",
                self.path.name,
                size,
                self.max_bytes,
            )
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            # Write to a temp file in the same directory, then swap it in, so an
            # interrupted write cannot leave a half-written state file behind.
            fd, tmp = tempfile.mkstemp(dir=str(DATA_DIR), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                os.replace(tmp, self.path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
        except OSError:
            log.exception("Could not write %s", self.path.name)
            return
        self._dirty = False
        self._last_save = time.monotonic()
