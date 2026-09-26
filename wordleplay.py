"""Preston plays today's Wordle - honestly.

The answer comes from the NYT, but the player never sees it: the code scores
each guess against it exactly as the game does and hands back only the
colours. The model guesses from that feedback, one word at a time.
"""

from __future__ import annotations

import re

GREEN, YELLOW, GREY = "G", "Y", "B"
EMOJI = {GREEN: "🟩", YELLOW: "🟨", GREY: "⬛"}
WORD_RE = re.compile(r"^[a-z]{5}$")


def score(guess: str, answer: str) -> list[str]:
    """Wordle's own colouring, duplicates included: greens first, then each
    remaining letter is yellow only while the answer still has one unclaimed."""
    guess, answer = guess.lower(), answer.lower()
    out = [GREY] * 5
    left: dict[str, int] = {}
    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            out[i] = GREEN
        else:
            left[a] = left.get(a, 0) + 1
    for i, g in enumerate(guess):
        if out[i] == GREEN:
            continue
        if left.get(g, 0) > 0:
            out[i] = YELLOW
            left[g] -= 1
    return out


def row(marks: list[str]) -> str:
    return "".join(EMOJI[m] for m in marks)


def knowledge(history: list[tuple[str, list[str]]]) -> str:
    """What the feedback so far proves, in plain words - small models forget
    their own greys, so the code keeps the ledger for them."""
    fixed = ["_"] * 5
    present: dict[str, set[int]] = {}
    absent: set[str] = set()
    for guess, marks in history:
        seen_here = {g for g, m in zip(guess, marks) if m != GREY}
        for i, (g, m) in enumerate(zip(guess, marks)):
            if m == GREEN:
                fixed[i] = g
            elif m == YELLOW:
                present.setdefault(g, set()).add(i + 1)
            elif g not in seen_here:
                absent.add(g)
    lines = [f"Pattern so far: {' '.join(c.upper() for c in fixed)}"]
    if present:
        lines.append("In the word but NOT at these positions: " + ", ".join(
            f"{k.upper()} (not {', '.join(map(str, sorted(v)))})" for k, v in sorted(present.items())))
    if absent:
        lines.append("Not in the word at all: " + " ".join(sorted(a.upper() for a in absent)))
    return "\n".join(lines)


def consistent(guess: str, history: list[tuple[str, list[str]]]) -> bool:
    """Would this guess have produced the same colours for every earlier guess?"""
    return all(score(prev, guess) == marks for prev, marks in history)
