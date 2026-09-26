"""The intent gates, pinned to the phrasings that have bitten before.

Runs under pytest or on its own:

    .venv\\Scripts\\python tests\\test_intents.py

Everything here is a pure function of the text - no Discord, no model. Each
case is (phrasing, expectation); the expectation names the path on_message
would take. When a gate regex changes, this is what says whether "paint the
calipers red or black?" still starts a render.
"""
from __future__ import annotations

import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot  # noqa: E402
import imagegen  # noqa: E402

Bot = next(v for v in vars(bot).values() if isinstance(v, type) and hasattr(v, "RAP_ABOUT_RE"))


# ---- image requests ---------------------------------------------------------

IMAGE_SURE = [
    "render an image of a mk7 gti on fire",
    "make me a picture of member_x grabbing a hoagie",
    "show me a picture of the simos 18 ecu",
    "gimme a quick meme of member_x",
    "I want a picture of a jetta on fire",
    "rendor me an image of a gti",
    "can you make this into a meme",
    "picture this",
    "render an image of this",
]
IMAGE_MAYBE = [                       # bare verb in command position -> router
    "draw this",
    "draw member_x grabbing a hoagie",
    "please draw a gti",
    "yo paint member_x as a clown",
    "paint my car purple",
    "paint the calipers red or black?",
    "i drew a picture of my car yesterday",
    "what does a picture of the simos 18 ecu look like",
]
IMAGE_NO = [
    "sketch out a plan for my build",
    "he painted his calipers red",
    "the render took forever",
    "make a song about member_x",
    "my paint is peeling",
    "how do i render the map in a2l",
    "make this a cartoon",             # edit verb; only counts with a picture attached
]


def test_image_confidence():
    for text in IMAGE_SURE:
        assert imagegen.image_confidence(text) == "sure", text
    for text in IMAGE_MAYBE:
        assert imagegen.image_confidence(text) == "maybe", text
    for text in IMAGE_NO:
        assert imagegen.image_confidence(text) == "", text


DEICTIC_SUBJECTS = [
    ("render an image of this", "this"),
    ("draw that", "that"),
    ("make a picture of what he said", "what he said"),
    ("can you make this into a meme", "this"),
    ("picture this", ""),
    ("draw me a picture", ""),
    ("illustrate his message", "his message"),
    ("draw the guy above", "the guy above"),
]


def test_deictic_subjects_are_degenerate():
    for text, want in DEICTIC_SUBJECTS:
        subject = imagegen.extract_prompt(text)
        assert subject == want, (text, subject)
        assert imagegen.is_degenerate(subject), text


def test_real_subjects_survive():
    for text, want in [
        ("render an image of a mk7 gti on fire", "a mk7 gti on fire"),
        ("make me a picture of member_x grabbing a hoagie", "member_x grabbing a hoagie"),
        ("gimme a quick meme of member_x", "member_x"),
    ]:
        subject = imagegen.extract_prompt(text)
        assert subject == want, (text, subject)
        assert not imagegen.is_degenerate(subject), text


def test_edit_needs_a_picture():
    for text in ["make this a cartoon", "put my car in the snow", "draw this but as an oil painting"]:
        assert imagegen.wants_edit(text), text
    assert not imagegen.wants_edit("draw member_x grabbing a hoagie")


def test_meme_flag():
    assert imagegen.wants_meme("create a meme of what member_x said")
    assert not imagegen.wants_meme("remember me")


# ---- songs, raps, styles ----------------------------------------------------

def parse_rap(text: str):
    """Mirror maybe_rap_about's parsing up to the member lookup."""
    m = Bot.RAP_ABOUT_RE.search(text.strip())
    if not m:
        return None
    who = m.group("who").strip(" .!?")
    who = re.split(r",|\.\s|\s+(?:but|then)\s+", who, maxsplit=1)[0]
    who = re.split(
        r"\s+and\s+(?=(?:type|write|post|put|keep|look|make|send|sing|just|no\b|don'?t|text))",
        who, maxsplit=1, flags=re.I,
    )[0].strip(" .!?")
    style = ""
    genre = (m.group("genre") or "").strip()
    if genre and genre.lower() not in Bot.GENRE_NOISE and not genre.isdigit():
        style = genre
    sm = Bot.STYLE_CLAUSE_RE.search(who)
    if sm:
        who = who[:sm.start()].strip(" .!?")
        style = (style + " " + sm.group("style").strip(" .!?\"'")).strip()
    st = Bot.STYLE_STATED_RE.search(text)
    if st and not style:
        style = st.group("style").strip(" .!?\"'")
    form = m.group("form").lower()
    form = {"diss track": "diss", "bars": "rap", "verse": "rap", "limericks": "limerick",
            "haikus": "haiku"}.get(form, form)
    return form, who, style, bool(Bot.SONG_TEXT_ONLY_RE.search(text)), bool(Bot.DEICTIC_WHO_RE.match(who))


RAP_CASES = [
    # text                                                       form    who               style          text_only deictic
    ("make a song about member_x",                                 ("song", "member_x", "", False, False)),
    ("write death metal song about member_f.",                      ("song", "member_f", "death metal", False, False)),
    ("make a country song about member_x",                         ("song", "member_x", "country", False, False)),
    ("sing a song about member_f in the style of johnny cash",      ("song", "member_f", "johnny cash", False, False)),
    ("make a song about member_x like a sea shanty",               ("song", "member_x", "a sea shanty", False, False)),
    ("make a song about the dyno day. style: yacht rock",        ("song", "the dyno day", "yacht rock", False, False)),
    ("write a song about the green names, look up when it was mentioned. the style is folk country",
                                                                 ("song", "the green names", "folk country", False, False)),
    ("make a song about pops and bangs",                         ("song", "pops and bangs", "", False, False)),
    ("make a song about me and type it here",                    ("song", "me", "", True, False)),
    ("write a song about member_a, just the lyrics",              ("song", "member_a", "", True, False)),
    ("make a death metal song about member_a and type it here",   ("song", "member_a", "death metal", True, False)),
    ("write a quick song about member_l",                           ("song", "member_l", "", False, False)),
    ("make a rap about member_b",                                 ("rap", "member_b", "", False, False)),
    ("do a poem about member_x in the style of dr seuss",          ("poem", "member_x", "dr seuss", False, False)),
    ("make me a diss track about member_x as gregorian chant",     ("diss", "member_x", "gregorian chant", False, False)),
    ("write a rap about him",                                    ("rap", "him", "", False, True)),
    ("make a song about this",                                   ("song", "this", "", False, True)),
    ("write a poem about the guy above",                         ("poem", "the guy above", "", False, True)),
    ("make a song about what he said",                           ("song", "what he said", "", False, True)),
]


def test_rap_parsing():
    for text, want in RAP_CASES:
        got = parse_rap(text)
        assert got == want, (text, got, want)


def test_rap_regex_leaves_chat_alone():
    for text in ["that song about the gti was great", "i wrote a poem once", "rap is dead"]:
        assert Bot.RAP_ABOUT_RE.search(text) is None, text


# ---- length and self-quote gates -------------------------------------------

def test_long_reply_gate():
    for text in ["explain this properly", "explain it fully", "walk me through it", "add more existential dread", "continue"]:
        assert bot.wants_long_reply(text), text
    for text in ["wtf lol", "is he right?", "what does he mean"]:
        assert not bot.wants_long_reply(text), text


def test_self_quote_is_ignored_by_gates():
    label = ("\n\n[Replying to YOUR OWN earlier message, quoted here so you know what they are "
             "reacting to. RESPOND TO WHAT THEY JUST SAID ABOVE - a jab ... (add more X, shorter, "
             "redo it) do you rework the quoted text: The relay coil collapses and currently ...]")
    asked = bot.strip_self_quote("lol ok" + label)
    assert asked == "lol ok"
    assert not bot.wants_long_reply(asked)
    assert not imagegen.wants_image(asked)


def test_summarize_this_is_not_a_channel_summary():
    # parse_summarize_intent still fires; on_message drops it when the target is a reply.
    count, about = bot.parse_summarize_intent("summarize this")
    assert re.fullmatch(r"(?:this|that|it|his|her|their|the) ?(?:message|comment|post|one)?", about.strip(), re.I)
    assert bot.parse_summarize_intent("summarize the last 50 messages about turbos") is not None


def test_plan_lists_are_caught():
    plan = ("1. Isolate the behavioral premise from reported trauma responses.\n"
            "2. Examine longitudinal tracking records.\n"
            "3. Measure administrative deviation in graduation rates.\n"
            "4. Paternal presence correlates with developmental stability.")
    assert bot.looks_like_plan(plan)
    content = ("1. The IS20 runs out of shaft speed at 22 psi.\n"
               "2. Your intake temps climb 15 degrees over a pull.\n"
               "3. Timing pulls four degrees by third gear.")
    assert not bot.looks_like_plan(content)
    assert not bot.looks_like_plan("Verify the harness. Then measure the rail.")
    assert not bot.looks_like_plan("- run it hot\n- blame the tune")


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
    sys.exit(1 if failed else 0)
