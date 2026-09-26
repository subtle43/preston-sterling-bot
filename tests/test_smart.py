"""The pure halves of the "smarter" work: the classifier's parser, the
follow-up matcher, the two-requests splitter, the web-search tiers, loose
member names, and the feedback maths. No Discord, no model.

    .venv\\Scripts\\python tests\\test_smart.py
"""
from __future__ import annotations

import os
import sys
import types
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chatsearch  # noqa: E402
import feedback  # noqa: E402
import intent  # noqa: E402
import websearch  # noqa: E402


# ---- intent.parse -----------------------------------------------------------

def test_parse_clean_json():
    got = intent.parse('{"intent":"image","subject":"a red mk7 golf on jack stands","style":"oil painting",'
                       '"audio":null,"length":"normal","needs_web":false,"needs_archive":true,'
                       '"text_only":false,"secondary":"song","secondary_subject":"a red mk7 golf on jack stands"}')
    assert got is not None and got.source == "model"
    assert got.intent == "image" and got.wants_picture
    assert got.subject == "a red mk7 golf on jack stands"
    assert got.style == "oil painting"
    assert got.audio is None and got.needs_web is False and got.needs_archive is True
    assert got.secondary == "song" and got.secondary_subject.startswith("a red")


def test_parse_tolerates_fences_and_prose():
    raw = 'Sure, here you go:\n```json\n{"intent": "song", "subject": "member_x", "audio": true, "length": "short"}\n```\nHope that helps.'
    got = intent.parse(raw)
    assert got is not None and got.intent == "song" and got.audio is True and got.length == "short"
    assert got.wants_piece


def test_parse_defaults_bad_fields():
    got = intent.parse('{"intent":"chat","length":"enormous","audio":"yes","needs_web":"no","secondary":"chat","subject":"none"}')
    assert got is not None
    assert got.length == "normal"
    assert got.audio is True and got.needs_web is False
    assert got.secondary == "" and got.secondary_subject == ""
    assert got.subject == ""


def test_parse_rejects_garbage():
    assert intent.parse("") is None
    assert intent.parse("YES") is None
    assert intent.parse('{"intent":"dance"}') is None
    assert intent.parse('{"intent": "image"') is None
    assert intent.parse("[1,2]") is None


def test_build_input_mentions_context():
    text = intent.build_input("draw this", reply_author="MEMBER_X", reply_text="my jetta is on jack stands",
                              has_image=False, reply_has_image=False, last_media="image of 'a gti' 3 min ago")
    assert "MEMBER_X" in text and "jack stands" in text and "3 min ago" in text
    assert "REPLYING TO: nothing" in intent.build_input("hi")


# ---- follow-ups ---------------------------------------------------------------

def test_followup_again():
    for text, want in [
        ("again", ("again", "")),
        ("again but as a cartoon", ("again", "as a cartoon")),
        ("the same again", ("again", "")),
        ("ok do it again, darker this time", ("again", "darker this time")),
        ("another one", ("again", "")),
        ("another one but in the snow", ("again", "in the snow")),
        ("one more time with a spoiler", ("again", "with a spoiler")),
        ("same but country", ("again", "country")),
        ("same song but country", ("again", "country")),
        ("redo it as an oil painting", ("again", "as an oil painting")),
        ("run it back", ("again", "")),
    ]:
        got = intent.parse_followup(text)
        assert got == want, (text, got, want)


def test_followup_edit():
    for text in ["make it darker", "now as a cartoon, redraw it", "put it in the snow", "add a spoiler to it",
                 "make the car red", "make this black and white", "turn it into a cartoon"]:
        got = intent.parse_followup(text)
        assert got is not None and got[0] == "edit", (text, got)


def test_followup_negatives():
    for text in ["make it stop", "again?", "another beer", "same", "make it make sense", "keep it up",
                 "make it quick", "i drew it again yesterday", "what happened to the same guy",
                 "turn it off", "change the subject"]:
        assert intent.parse_followup(text) is None, text


# ---- two requests -------------------------------------------------------------

def test_split_requests():
    assert intent.split_requests("draw member_x's car and write a song about it") == \
        ["draw member_x's car", "write a song about it"]
    assert intent.split_requests("make a song about X and then draw the cover") == \
        ["make a song about X", "draw the cover"]
    assert intent.split_requests("render a gti and a jetta") == ["render a gti and a jetta"]
    assert intent.split_requests("make a song about pops and bangs") == ["make a song about pops and bangs"]
    assert intent.split_requests("make a song about me and type it here") == ["make a song about me and type it here"]
    assert intent.split_requests("") == []


def test_resolve_pronoun():
    assert intent.resolve_pronoun("write a song about it", "member_x's car") == "write a song about member_x's car"
    assert intent.resolve_pronoun("draw that", "the dyno day") == "draw the dyno day"
    assert intent.resolve_pronoun("write a song about member_f", "x") == "write a song about member_f"
    assert intent.resolve_pronoun("draw it", "") == "draw it"


# ---- web search tiers ---------------------------------------------------------

def test_search_confidence():
    for text in ["look it up", "search for the new golf r specs", "google it", "check online what a 2.0t weighs",
                 "any news on the mk9 golf"]:
        assert websearch.search_confidence(text) == "sure", text
    for text in ["currently running 22 psi, is that ok", "what is the score for this map", "who won the f1 race",
                 "who is the new vw ceo", "what's the price of a is38 these days", "weather tomorrow"]:
        assert websearch.search_confidence(text) == "maybe", text
    for text in ["my file is 100% mine", "wtf lol", "how does the wastegate work", ""]:
        assert websearch.search_confidence(text) == "", text
    assert websearch.wants_search("look it up") and not websearch.wants_search("wtf lol")


def test_personal_info_still_refused():
    assert websearch.asks_for_personal_info("look up his home address")


# ---- loose member names -------------------------------------------------------

def _fake_index(names):
    stub = types.SimpleNamespace()
    stub.speakers = {n: [0] * rows for n, rows in names.items()}
    stub.aliases = {}
    pool = sorted((n for n, rows in names.items() if rows >= chatsearch.FUZZY_MIN_ROWS), key=len, reverse=True)
    stub._fuzzy_plain = [(n, "".join(ch for ch in n if ch.isalnum())) for n in pool]
    stub._rank_names = lambda name: chatsearch.ChatIndex._rank_names(stub, name)
    stub.resolve_loose = lambda name: chatsearch.ChatIndex.resolve_loose(stub, name)
    stub.suggest = lambda name, limit=3: chatsearch.ChatIndex.suggest(stub, name, limit)
    return stub


def test_resolve_loose():
    idx = _fake_index({
        "mrturbo [anti-clanker]": 900, "danny7": 1400, "danny": 3, "wrench9": 800, "gti4life": 2000,
        "the original gremlins": 300, "numlock": 400, "sleepy": 600, "sleepyhead9": 120,
    })
    assert idx.resolve_loose("mrturbo [anti-clanker]") == "mrturbo [anti-clanker]"   # exact
    assert idx.resolve_loose("turbo") == "mrturbo [anti-clanker]"                     # inside the handle
    assert idx.resolve_loose("clanker") == "mrturbo [anti-clanker]"                   # a bracket word
    assert idx.resolve_loose("danny") == "danny7"                                     # prefix, regular wins
    assert idx.resolve_loose("gremlins") == "the original gremlins"                   # a word of the handle
    assert idx.resolve_loose("nummlock") == "numlock"                                 # near miss
    assert idx.resolve_loose("gti4life") == "gti4life"
    assert idx.resolve_loose("zz") is None and idx.resolve_loose("") is None
    assert idx.resolve_loose("nobody-like-this") is None
    # "sleepy" is exact for one regular even though another starts with it.
    assert idx.resolve_loose("sleepy") == "sleepy"
    # Two regulars that both start with the typed name: ambiguous, suggest both.
    idx2 = _fake_index({"danny7": 1400, "dannyb": 1300, "gti4life": 2000})
    assert idx2.resolve_loose("dann") is None
    assert set(idx2.suggest("dann", 2)) == {"danny7", "dannyb"}


# ---- feedback maths -----------------------------------------------------------

class _Store:
    def __init__(self):
        self.data = {}
        self.saved = 0
    def load(self): return self.data
    def touch(self): pass
    def maybe_save(self): self.saved += 1
    def save(self): self.saved += 1


def test_feedback_empty_is_flat():
    fb = feedback.Feedback(_Store())
    assert fb.multipliers(1, ["none", "s1", "s2"]) == {"none": 1.0, "s1": 1.0, "s2": 1.0}


def test_feedback_scoring_and_clamp():
    fb = feedback.Feedback(_Store(), keep=3)
    for i in range(20):
        fb.note_posted(100 + i, guild_id=1, shape="s1")
    for i in range(20):
        fb.note_posted(200 + i, guild_id=1, shape="s2")
    for i in range(20):
        fb.note_posted(300 + i, guild_id=1, shape="none")
    # The bounded map only remembers the last 3 messages posted.
    assert len(fb.posted) == 3 and 100 not in fb.posted
    # Reactions on messages it no longer remembers are ignored.
    assert fb.note_reaction(100, 7, "😂") is False
    # Every remembered "none" reply gets a laugh; s1/s2 none.
    for mid in list(fb.posted):
        assert fb.note_reaction(mid, 7, "😂") is True
    mults = fb.multipliers(1, ["none", "s1", "s2"])
    assert mults["none"] > 1.0 >= mults["s1"] == mults["s2"]
    assert all(0.5 <= v <= 2.0 for v in mults.values())
    # Dedupe: the same person reacting twice is one reaction, one laugh.
    mid = next(iter(fb.posted))
    row = fb._row(1, "none")
    before = (row["reactions"], row["laughs"])
    fb.note_reaction(mid, 7, "🤣")
    assert (row["reactions"], row["laughs"]) == before
    # Removing the reactions takes them back off.
    fb.note_reaction(mid, 7, "🤣", added=False)
    fb.note_reaction(mid, 7, "😂", added=False)
    assert (row["reactions"], row["laughs"]) == (before[0] - 1, before[1] - 1)


def test_feedback_cap_and_laugh_classifier():
    fb = feedback.Feedback(_Store())
    fb.note_posted(1, guild_id=9, shape="s3")
    for user in range(30):
        fb.note_reaction(1, user, "💀")
    for i in range(50):
        fb.note_posted(10 + i, guild_id=9, shape="s1")
        fb.note_posted(100 + i, guild_id=9, shape="s2")
    # One shape that always lands against two that never do: capped at 2.0
    # and floored at 0.5, never a runaway.
    mults = fb.multipliers(9, ["s1", "s2", "s3"])
    assert mults["s3"] == 2.0 and mults["s1"] == 0.5 and mults["s2"] == 0.5
    assert feedback.is_laugh("😂") and feedback.is_laugh("", "kekw") and feedback.is_laugh("<:pepelaugh:1>", "pepelaugh")
    assert not feedback.is_laugh("👍") and not feedback.is_laugh("🚗", "car")
    rows = fb.table(9, ["s1", "s3"])
    assert [r["shape"] for r in rows][:2] == ["s1", "s3"] and rows[1]["laughs"] == 30


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL  {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)
