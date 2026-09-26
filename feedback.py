"""What lands. The room reacts to the replies it likes; this keeps score.

Each reply the bot posts is filed under the shape it was written in (one of
REPLY_SHAPES, or "none", or the kind of thing it was: roast, song, image).
Reactions from other people on those messages are counted against that
shape - any emoji as a reaction, the laughing ones as a laugh. The shape roll
then leans toward what this room actually laughs at instead of a fixed table.

Counts persist per guild in data/feedback.json. Which message used which shape
lives only in memory, bounded, because a reaction on a reply from last week
is noise.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict

log = logging.getLogger("ollama-discord")

# Reactions that mean "that was funny", as opposed to a thumbs-up or a car.
LAUGH_EMOJI = frozenset("😂🤣💀😭😆😹☠️🫠🤭😅🔥💯👏")
LAUGH_NAMES = ("joy", "rofl", "skull", "sob", "laugh", "lmao", "kek", "lul", "lol", "dead", "crying", "fire", "100")

# Bayesian smoothing: a shape with one laugh in one use is not twice as good as
# one with ten in twenty. The prior is worth twenty-five uses at the table's
# mean, and nothing moves at all until the shapes have been used fifty times
# between them. It was four and zero: after one evening - eighteen plain
# replies, two laughs - plain replies were at half weight and the mocking
# shapes up a third, which is a personality change decided by noise.
PRIOR_USES = 25
MIN_TOTAL_USES = 50
MULT_MIN, MULT_MAX = 0.5, 2.0


def is_laugh(emoji: str, name: str = "") -> bool:
    text = (emoji or "").strip()
    if text and text in LAUGH_EMOJI:
        return True
    low = (name or "").lower()
    return bool(low) and any(part in low for part in LAUGH_NAMES)


class Feedback:
    def __init__(self, store, *, keep: int = 500) -> None:
        self.store = store
        self.store.load()
        self.keep = keep
        # message id -> what it was, plus who has reacted so far (per user, so
        # one person mashing five emoji is one reaction).
        self.posted: OrderedDict[int, dict] = OrderedDict()
        self._logged_at = 0.0
        self._logged: dict[str, float] = {}

    # ---- storage -----------------------------------------------------------

    def _table(self, guild_id: int) -> dict[str, dict]:
        guilds = self.store.data.setdefault("guilds", {})
        return guilds.setdefault(str(int(guild_id)), {})

    def _row(self, guild_id: int, shape: str) -> dict:
        return self._table(guild_id).setdefault(shape, {"uses": 0, "reactions": 0, "laughs": 0})

    def _save(self) -> None:
        self.store.touch()
        self.store.maybe_save()

    # ---- events ------------------------------------------------------------

    def note_posted(self, message_id: int, *, guild_id: int, shape: str, kind: str = "reply") -> None:
        """A reply went up. Uses are counted here, at post time - otherwise a
        shape nobody reacts to never registers and looks untested forever."""
        self.posted[int(message_id)] = {
            "guild": int(guild_id), "shape": shape, "kind": kind, "ts": time.time(), "reactors": {},
        }
        while len(self.posted) > self.keep:
            self.posted.popitem(last=False)
        self._row(guild_id, shape)["uses"] += 1
        self._save()

    def note_reaction(self, message_id: int, user_id: int, emoji: str, name: str = "", *, added: bool = True) -> bool:
        """One of ours got (or lost) a reaction. True if it was one we track."""
        entry = self.posted.get(int(message_id))
        if entry is None:
            return False
        reactors: dict = entry["reactors"]
        theirs: set = reactors.setdefault(int(user_id), set())
        row = self._row(entry["guild"], entry["shape"])
        laugh = is_laugh(emoji, name)
        if added:
            if emoji in theirs:
                return True
            first = not theirs
            theirs.add(emoji)
            if first:
                row["reactions"] += 1
            if laugh and not any(is_laugh(e) for e in theirs if e != emoji):
                row["laughs"] += 1
        else:
            if emoji not in theirs:
                return True
            theirs.discard(emoji)
            if not theirs:
                row["reactions"] = max(0, row["reactions"] - 1)
            if laugh and not any(is_laugh(e) for e in theirs):
                row["laughs"] = max(0, row["laughs"] - 1)
        self._save()
        return True

    # ---- what to do with it --------------------------------------------------

    def multipliers(self, guild_id: int, shapes: list[str]) -> dict[str, float]:
        """Weight multipliers for the shape roll, 0.5-2.0, 1.0 with no data.

        score = (laughs + 1) / (uses + PRIOR_USES), relative to the mean score
        across the shapes offered, so a table where nothing ever gets a laugh
        stays flat rather than sinking everything to the floor.
        """
        table = self._table(guild_id) if guild_id else {}
        if sum(table.get(s, {}).get("uses", 0) for s in shapes) < MIN_TOTAL_USES:
            return {s: 1.0 for s in shapes}
        scores = {
            s: (table.get(s, {}).get("laughs", 0) + 1) / (table.get(s, {}).get("uses", 0) + PRIOR_USES)
            for s in shapes
        }
        mean = sum(scores.values()) / max(1, len(scores))
        result = {
            s: max(MULT_MIN, min(MULT_MAX, score / mean)) if mean > 0 else 1.0
            for s, score in scores.items()
        }
        self._maybe_log(guild_id, result)
        return result

    def _maybe_log(self, guild_id: int, mults: dict[str, float]) -> None:
        now = time.time()
        moved = any(abs(mults.get(k, 1.0) - self._logged.get(k, 1.0)) >= 0.25 for k in mults)
        if now - self._logged_at < 3600 and not moved:
            return
        self._logged_at, self._logged = now, dict(mults)
        log.info("Shape weights for guild %s: %s", guild_id,
                 ", ".join(f"{k}x{v:.2f}" for k, v in sorted(mults.items())))

    def table(self, guild_id: int, shapes: list[str] | None = None) -> list[dict]:
        """Rows for "!feedback": shape, uses, reactions, laughs, multiplier."""
        stored = self._table(guild_id)
        keys = list(shapes or []) + [k for k in stored if k not in (shapes or [])]
        mults = self.multipliers(guild_id, keys) if keys else {}
        rows = []
        for key in keys:
            row = stored.get(key, {})
            rows.append({
                "shape": key, "uses": row.get("uses", 0), "reactions": row.get("reactions", 0),
                "laughs": row.get("laughs", 0), "multiplier": mults.get(key, 1.0),
            })
        return rows
