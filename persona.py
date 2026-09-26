"""Preston's personality, swappable without touching code.

Each persona is a folder under prompts/personas/<name>/ with three files:

    system.txt      the full persona (the full harness reads this)
    lite.txt        the short version (lite harness and every slash-command gag)
    voice_note.txt  one paragraph appended to each reply - the model obeys what
                    it reads last, so this is what keeps the voice on track

The active one is kept in data/persona.json and switched live with /persona.
Everything else in the bot is written persona-neutral ("in your persona's
voice"), so adding a personality is: copy a folder, rewrite three text files,
/persona <name>. Files are read fresh on every use, so editing one takes
effect on the next reply.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PERSONAS = ROOT / "prompts" / "personas"
STATE = ROOT / "data" / "persona.json"
DEFAULT = "therapist"
KINDS = ("system", "lite", "voice_note")


SHARED_RULES = PERSONAS / "_shared" / "rules.txt"
RULES_MARKER = "THE RULE ABOVE ALL OTHERS"


def available() -> list[str]:
    if not PERSONAS.is_dir():
        return []
    return sorted(p.name for p in PERSONAS.iterdir()
                  if p.is_dir() and not p.name.startswith("_") and (p / "system.txt").exists())


def active() -> str:
    try:
        name = json.loads(STATE.read_text(encoding="utf-8")).get("persona", "")
    except (OSError, ValueError):
        name = ""
    names = available()
    if name in names:
        return name
    return DEFAULT if DEFAULT in names else (names[0] if names else "")


def set_active(name: str) -> bool:
    if name not in available():
        return False
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"persona": name}), encoding="utf-8")
    return True


def about(name: str) -> str:
    """The persona's one-line description (about.txt), or "" if it has none."""
    try:
        return (PERSONAS / name / "about.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def flags(name: str | None = None) -> set[str]:
    """Optional switches in flags.txt, one per line:

        big_words   gets the weekly handful of precise words, and drunk mode keeps
                    the "vocabulary goes up" rule (the classic engineer voice)
        no_drunk    never rolls the drunk mood (for characters it doesn't fit)
    """
    try:
        text = (PERSONAS / (name or active()) / "flags.txt").read_text(encoding="utf-8")
    except OSError:
        return set()
    return {line.strip().lower() for line in text.splitlines() if line.strip() and not line.startswith("#")}


def song_styles(name: str | None = None) -> list[str]:
    """The persona's music genres (song_style.txt, one 'genre: description' per
    line). Empty means songs use the bot's random style list."""
    try:
        text = (PERSONAS / (name or active()) / "song_style.txt").read_text(encoding="utf-8")
    except OSError:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def image_style(name: str | None = None) -> str:
    """The persona's default picture look (image_style.txt), or "" for none."""
    try:
        return (PERSONAS / (name or active()) / "image_style.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def read(kind: str, name: str | None = None) -> str:
    """One of the active persona's files, or "" if it has none (callers fall
    back to their own default)."""
    if kind not in KINDS:
        raise ValueError(kind)
    name = name or active()
    if not name:
        return ""
    try:
        text = (PERSONAS / name / f"{kind}.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    # A persona's system.txt only needs its CHARACTER section: the shared rules
    # (answer first, never invent labels, honesty, privacy, capabilities) are
    # appended from _shared/rules.txt unless the file carries its own copy.
    if kind == "system" and text and RULES_MARKER not in text:
        try:
            text = text + "\n\n" + SHARED_RULES.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return text
