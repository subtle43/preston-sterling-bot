"""Phase 1 of the server chat index: pull the history to disk. Read-only.

Deliberately separate from the indexing step. The crawl is bound by Discord's
history rate limit at roughly 97 messages a second and cannot be made faster; the
embedding pass runs at about 44 chunks a second and CAN be redone. Keeping them
apart means changing the chunking strategy is a two-minute rebuild rather than
another multi-hour crawl.

Resumable per channel. Progress is written after every batch, so an interrupted
run - or a rate-limit stall, or closing the laptop - costs at most one batch
rather than the whole crawl. Re-running picks up from the last message seen in
each channel and adds only what is new.

    python scrape_chat.py              # resume (or start) the crawl
    python scrape_chat.py --restart    # ignore progress and pull everything again
    python scrape_chat.py --status     # what has been collected so far

Nothing here talks to a model. It writes raw.jsonl and state.json and stops.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path

import discord

from config import Settings

log = logging.getLogger("scrape-chat")

# Messages shorter than this carry nothing a search could ever want. Measured on
# this server the mean message is 59 characters, so a large share of the volume is
# "lol", "yeah", "same" - dropping them cuts embedding time AND sharpens
# retrieval, because noise dilutes the vector of whatever chunk it lands in.
MIN_CHARS = 15

# Written every N messages within a channel so a kill costs one batch at most.
CHECKPOINT_EVERY = 500

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# A message that is only a link is a pointer, not content.
_ONLY_URL_RE = re.compile(r"^\s*(?:https?://\S+\s*)+$", re.I)


def clean(text: str) -> str:
    """Flatten to one safe line. Same reasoning as lore.clean_fact.

    Newlines go so a stored message cannot later forge a heading and pose as an
    instruction once it is injected into a prompt as retrieved context.
    """
    return " ".join(_CONTROL_RE.sub(" ", text or "").split())


def worth_keeping(message: discord.Message) -> bool:
    if message.author.bot:
        return False
    body = clean(message.content)
    if len(body) < MIN_CHARS:
        return False
    if _ONLY_URL_RE.match(body):
        return False
    return True


class Scraper:
    def __init__(self, settings: Settings, restart: bool, max_per_channel: int = 0) -> None:
        self.settings = settings
        self.restart = restart
        # 0 means no cap. With a cap, a channel stops after this many messages in
        # THIS run and its cursor is saved as normal - so a later run continues
        # from exactly where it left off rather than re-reading. Three channels
        # held 68% of the remaining crawl time; capping them buys breadth across
        # every channel now and leaves the depth available later.
        self.max_per_channel = max(0, int(max_per_channel))
        self.dir = Path(settings.chat_index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.dir / "raw.jsonl"
        self.state_path = self.dir / "state.json"
        self.state: dict[str, dict] = {}
        if self.state_path.exists() and not restart:
            try:
                self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                log.warning("state.json unreadable - starting fresh")
        self.kept = 0
        self.seen = 0
        self.started = time.monotonic()

    def save_state(self) -> None:
        self.state_path.write_text(
            json.dumps(self.state, indent=1), encoding="utf-8"
        )

    async def scrape_channel(self, channel, out) -> None:
        key = str(channel.id)
        prior = self.state.get(key) or {}
        after_id = prior.get("last_id")
        # oldest_first with `after` is what makes resume exact: the cursor is the
        # newest message already stored, so nothing is re-read and nothing skipped.
        after = discord.Object(id=int(after_id)) if after_id else None

        n_seen = n_kept = 0
        last_id = after_id
        t0 = time.monotonic()
        try:
            async for msg in channel.history(limit=None, oldest_first=True, after=after):
                n_seen += 1
                last_id = msg.id
                if worth_keeping(msg):
                    out.write(json.dumps({
                        "id": msg.id,
                        "channel_id": channel.id,
                        "channel": channel.name,
                        "author_id": msg.author.id,
                        "author": msg.author.display_name,
                        "ts": int(msg.created_at.timestamp()),
                        "text": clean(msg.content),
                    }, ensure_ascii=False) + "\n")
                    n_kept += 1
                if self.max_per_channel and n_seen >= self.max_per_channel:
                    print(f"    capped at {self.max_per_channel:,} this run "
                          f"(resume later to continue)", flush=True)
                    break
                if n_seen % CHECKPOINT_EVERY == 0:
                    out.flush()
                    self.state[key] = {
                        "name": channel.name, "last_id": last_id,
                        "seen": prior.get("seen", 0) + n_seen,
                        "kept": prior.get("kept", 0) + n_kept,
                    }
                    self.save_state()
                    rate = n_seen / max(time.monotonic() - t0, 0.01)
                    print(f"    ...{n_seen:>7,} read, {n_kept:>7,} kept "
                          f"({rate:,.0f}/s)", flush=True)
        except discord.Forbidden:
            print(f"    no access partway through - keeping what we have", flush=True)
        except Exception as exc:
            # One bad channel must not cost the run. Its cursor is already saved.
            print(f"    ERROR {type(exc).__name__}: {str(exc)[:60]}", flush=True)

        out.flush()
        self.state[key] = {
            "name": channel.name, "last_id": last_id,
            "seen": prior.get("seen", 0) + n_seen,
            "kept": prior.get("kept", 0) + n_kept,
        }
        self.save_state()
        self.seen += n_seen
        self.kept += n_kept
        if n_seen:
            print(f"    {n_seen:>7,} read  {n_kept:>7,} kept  "
                  f"{time.monotonic() - t0:>5.0f}s", flush=True)

    async def run(self, client: discord.Client) -> None:
        mode = "w" if self.restart else "a"
        if self.restart and self.raw_path.exists():
            self.raw_path.unlink()
        excluded = set(self.settings.chat_index_exclude)
        with self.raw_path.open(mode, encoding="utf-8") as out:
            for guild in client.guilds:
                targets = []
                for ch in guild.text_channels:
                    targets.append(ch)
                    # Active threads hang off a channel and hold real conversation.
                    # Archived ones are skipped: another paginated call per channel
                    # for material nobody has touched in months.
                    targets.extend(getattr(ch, "threads", []) or [])
                print(f"\nGUILD {guild.name}: {len(targets)} channels/threads",
                      flush=True)
                for i, ch in enumerate(targets, 1):
                    parent_id = getattr(ch, "parent_id", None)
                    if ch.id in excluded or (parent_id and parent_id in excluded):
                        print(f"  [{i}/{len(targets)}] {ch.name} - EXCLUDED", flush=True)
                        continue
                    perms = ch.permissions_for(guild.me)
                    if not perms.read_message_history:
                        continue
                    done = (self.state.get(str(ch.id)) or {}).get("seen", 0)
                    print(f"  [{i}/{len(targets)}] {ch.name}"
                          f"{f' (resuming, {done:,} done)' if done else ''}", flush=True)
                    await self.scrape_channel(ch, out)
        elapsed = time.monotonic() - self.started
        print(f"\nread {self.seen:,} messages, kept {self.kept:,} "
              f"({self.kept / max(self.seen, 1):.0%}) in {elapsed / 60:.1f} min")
        print(f"-> {self.raw_path}")


def show_status(settings: Settings) -> None:
    d = Path(settings.chat_index_dir)
    state_path, raw_path = d / "state.json", d / "raw.jsonl"
    if not state_path.exists():
        print("Nothing scraped yet.")
        return
    state = json.loads(state_path.read_text(encoding="utf-8"))
    seen = sum(v.get("seen", 0) for v in state.values())
    kept = sum(v.get("kept", 0) for v in state.values())
    size = raw_path.stat().st_size / 1e6 if raw_path.exists() else 0
    print(f"channels touched : {len(state)}")
    print(f"messages read    : {seen:,}")
    print(f"messages kept    : {kept:,} ({kept / max(seen, 1):.0%})")
    print(f"raw.jsonl        : {size:,.1f} MB")
    top = sorted(state.values(), key=lambda v: v.get("kept", 0), reverse=True)[:10]
    print("\nbusiest channels:")
    for v in top:
        print(f"  {str(v.get('name'))[:28]:28} {v.get('kept', 0):>8,} kept")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", action="store_true",
                    help="ignore saved progress and re-pull everything")
    ap.add_argument("--status", action="store_true",
                    help="show what has been collected, then exit")
    ap.add_argument("--max-per-channel", type=int, default=0, metavar="N",
                    help="stop each channel after N messages this run (0 = no cap). "
                         "Cursors are still saved, so a later run continues.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    settings = Settings.load()
    if args.status:
        show_status(settings)
        return
    if not settings.discord_token:
        sys.exit("DISCORD_TOKEN is not set.")

    print(f"index dir : {settings.chat_index_dir}")
    print(f"excluded  : {sorted(settings.chat_index_exclude) or 'none'}")
    print(f"min chars : {MIN_CHARS}")
    if args.max_per_channel:
        print(f"cap       : {args.max_per_channel:,} messages per channel this run")

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    scraper = Scraper(settings, args.restart, args.max_per_channel)

    @client.event
    async def on_ready() -> None:
        try:
            await scraper.run(client)
        finally:
            await client.close()

    client.run(settings.discord_token, log_handler=None)


if __name__ == "__main__":
    main()
