"""One reading of what a message addressed to the bot wants.

Every path used to be its own regex - image, edit, meme, song, summarize, long
answer, web search, archive - and each phrasing that slipped past one of them
got patched into that one regex. This module puts a single small model call in
front of the gates: the message, what it replied to, whether a picture is in
play and what the bot last made in the channel go in; a JSON verdict comes out.
The regexes stay as the fast path for the obvious forms and as the whole path
when there is no judge (a local model, a timeout, a parse failure), so nothing
that worked before depends on this.

Also home to the two pure pieces of "smartness" that need no model at all: the
follow-up matcher ("again but as a cartoon", "make it darker") and the splitter
for two requests in one message ("draw X and write a song about it").
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass

log = logging.getLogger("ollama-discord")

INTENTS = (
    "image", "edit", "meme", "song", "rap", "poem", "other_creative",
    "summarize", "question", "chat",
)
PICTURE = frozenset({"image", "edit", "meme"})
PIECE = frozenset({"song", "rap", "poem", "other_creative"})
LENGTHS = ("short", "normal", "long")


@dataclass
class Intent:
    intent: str = "chat"
    subject: str = ""
    style: str = ""
    audio: bool | None = None
    length: str = "normal"
    needs_web: bool | None = None
    needs_archive: bool | None = None
    text_only: bool = False
    secondary: str = ""
    secondary_subject: str = ""
    source: str = "none"         # "model" when a judge produced it

    @property
    def wants_picture(self) -> bool:
        return self.intent in PICTURE

    @property
    def wants_piece(self) -> bool:
        return self.intent in PIECE

    def describe(self) -> str:
        bits = [f"intent={self.intent}"]
        if self.subject:
            bits.append(f"subject={self.subject[:50]!r}")
        if self.style:
            bits.append(f"style={self.style[:30]!r}")
        if self.audio is not None:
            bits.append(f"audio={self.audio}")
        if self.text_only:
            bits.append("text_only")
        bits.append(f"len={self.length}")
        bits.append(f"web={self.needs_web} archive={self.needs_archive}")
        if self.secondary:
            bits.append(f"then={self.secondary}:{self.secondary_subject[:30]!r}")
        bits.append(f"src={self.source}")
        return " ".join(bits)


SYSTEM = """You are the intent switch for "Preston Sterling", a Discord bot in a car-tuning server. The bot can: chat; answer questions (with a web search, or the server's own chat archive, when needed); review datalogs; generate pictures and memes; edit a picture it was shown; and write and SING songs (raps and poems too) about members or topics.

Given ONE message addressed to the bot, output ONLY a JSON object with these keys:
"intent": one of image | edit | meme | song | rap | poem | other_creative | summarize | question | chat
"subject": what to draw / sing about / write about. Resolve pointers: if they say "this", "that", "it", "him", "what he said" and a replied-to message is given, put that message's gist here (or its author's name for "him"/"her"). Empty string if there is no subject.
"style": a requested visual or musical style ("country", "in the style of johnny cash", "oil painting"), else ""
"audio": true if a song should be sung out loud, false if they want it typed, null if this is not a song
"length": "long" only when they ask to explain properly / in depth / walk through / more detail / continue; "short" for one-liners like "shorter"; else "normal"
"needs_web": true if a good answer needs current information from the internet (news, prices, weather, release dates, who won, recent events, anything after 2024)
"needs_archive": true if a good answer needs what members of THIS server have said or done (their cars, past posts, arguments, who said what, server history, a member by name)
"text_only": true if they say type it here / just the lyrics / text only / no audio / write it out
"secondary": a SECOND, different request in the same message ("draw X and write a song about it") using the same intent names, else ""
"secondary_subject": its subject with it/that resolved to the first subject, else ""

Rules: "edit" only when a picture is attached or replied to AND they ask to change it (make it darker, put it in the snow, add a spoiler, as a cartoon). A caption, joke, comment or question posted with a picture is chat or question, never edit. Talking ABOUT painting, rendering or drawing (paint on a car, a render they saw, something they drew yesterday) is chat or question, not image. A greeting, a jab, a reaction or banter is chat. A tuning question is question. A request to summarize the channel is summarize; "summarize this" under a replied-to message is question. Never invent a subject. Output the JSON and nothing else."""


# The same contract as an API schema, for backends that can enforce one
# (Gemini's responseSchema). Enums here mean parse() never sees a stray value.
SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "intent": {"type": "STRING", "enum": list(INTENTS)},
        "subject": {"type": "STRING"},
        "style": {"type": "STRING"},
        "audio": {"type": "BOOLEAN", "nullable": True},
        "length": {"type": "STRING", "enum": list(LENGTHS)},
        "needs_web": {"type": "BOOLEAN"},
        "needs_archive": {"type": "BOOLEAN"},
        "text_only": {"type": "BOOLEAN"},
        "secondary": {"type": "STRING"},
        "secondary_subject": {"type": "STRING"},
    },
    "required": ["intent", "subject", "style", "length", "needs_web", "needs_archive", "text_only"],
    "propertyOrdering": [
        "intent", "subject", "style", "audio", "length", "needs_web", "needs_archive",
        "text_only", "secondary", "secondary_subject",
    ],
}


def build_input(
    text: str, *, reply_author: str = "", reply_is_bot: bool = False, reply_text: str = "",
    has_image: bool = False, reply_has_image: bool = False, last_media: str = "",
) -> str:
    lines = [f"MESSAGE: {text.strip()[:600]}"]
    if reply_text or reply_author:
        who = "the bot (you)" if reply_is_bot else (reply_author or "someone")
        lines.append(f"REPLYING TO a message from {who}: {reply_text.strip()[:300]!r}")
    else:
        lines.append("REPLYING TO: nothing")
    lines.append(f"PICTURE ATTACHED: {'yes' if has_image else 'no'}; "
                 f"REPLIED MESSAGE HAS A PICTURE: {'yes' if reply_has_image else 'no'}")
    lines.append(f"LAST THING THE BOT MADE IN THIS CHANNEL: {last_media or 'nothing recently'}")
    return "\n".join(lines)


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)


def _as_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
    return None


def parse(raw: str) -> Intent | None:
    """The model's text -> Intent, or None when nothing usable came back.

    Defensive on purpose: fences, prose around the object, a wrong enum, a
    missing key or a string where a bool was asked for all degrade to the
    field's default rather than throwing the whole verdict away.
    """
    text = _FENCE_RE.sub("", raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None

    def clean_str(key: str, limit: int) -> str:
        value = data.get(key)
        if not isinstance(value, str):
            return ""
        return re.sub(r"\s+", " ", value).strip(" .\"'")[:limit]

    intent = clean_str("intent", 30).lower().replace(" ", "_")
    if intent not in INTENTS:
        return None
    length = clean_str("length", 10).lower()
    if length not in LENGTHS:
        length = "normal"
    secondary = clean_str("secondary", 30).lower().replace(" ", "_")
    if secondary not in INTENTS or secondary == intent:
        secondary = ""
    subject = clean_str("subject", 300)
    if subject.lower() in ("none", "null", "n/a", "-"):
        subject = ""
    return Intent(
        intent=intent,
        subject=subject,
        style=clean_str("style", 80),
        audio=_as_bool(data.get("audio")),
        length=length,
        needs_web=_as_bool(data.get("needs_web")),
        needs_archive=_as_bool(data.get("needs_archive")),
        text_only=bool(_as_bool(data.get("text_only"))),
        secondary=secondary,
        secondary_subject=clean_str("secondary_subject", 300) if secondary else "",
        source="model",
    )


async def classify(ollama, text: str, *, timeout: float = 4.0, **context) -> Intent | None:
    """Ask the judge. Raises nothing: a timeout or a bad reply is None.

    `ollama` is the bot's chat client (Gemini in practice - the caller decides
    whether the configured model is a judge worth waiting on).
    """
    if not (text or "").strip():
        return None
    messages = ollama.build_messages(SYSTEM, [], build_input(text, **context))
    # A judge call: minimal deliberation (the operator's thinking level is for
    # replies, not for a hundred tokens of JSON) and, where the backend can
    # enforce it, the schema itself.
    extra = {"response_schema": SCHEMA} if getattr(ollama, "supports_json_schema", False) else {}
    out = ""
    try:
        async with asyncio.timeout(timeout):
            async for delta, _ in ollama.stream_chat(
                messages, think="minimal", num_predict=220, temperature=0.0, **extra
            ):
                out += delta
    except Exception as exc:
        log.info("intent classifier unavailable (%s)", type(exc).__name__)
        return None
    result = parse(out)
    if result is None:
        log.info("intent classifier returned nothing usable: %r", out[:80])
    return result


# ---- follow-ups: "again", "make it darker" ----------------------------------

# "again", "another one", "same but country", "redo it" - do the last thing
# over. The rest of the line is the change they want. Bare "same" or "another
# <noun>" is conversation ("same", "another beer") and stays out.
AGAIN_RE = re.compile(
    r"^\s*(?:(?:ok|okay|now|and|please|pls|hey|yo|lol)[\s,]+)*"
    r"(?P<verb>again|do (?:it|that) again|one more(?: time)?|once more|try again|run it back|"
    r"redo(?: it| that)?|another(?: one)?(?=\s*$|\s+(?:but|with|in|as|of|like)\b|,)|"
    r"(?:the )?same(?: (?:song|one|thing|pic|picture|image|track|deal))?(?=\s+(?:but|again|with|in|as)\b))"
    r"(?P<rest>.*)$",
    re.I | re.S,
)
# Only the connective goes; "with a spoiler" and "as a cartoon" read fine
# appended to a subject, "a spoiler" does not.
_REST_LEAD_RE = re.compile(r"^[\s,.\-]*(?:but|and|this time|make it|but make it)?\s*", re.I)

# Visual vocabulary that makes "make it X" an edit of the last picture rather
# than an idiom ("make it stop", "make it quick").
_VISUAL_RE = re.compile(
    r"\b(dark(?:er)?|light(?:er)?|bright(?:er)?|blur+y|sharp(?:er)?|bigger|smaller|wider|closer|"
    r"cartoon\w*|anime|paint\w*|oil|watercolou?r|sketch\w*|pencil|drawing|illustrat\w*|pixel\w*|"
    r"retro|vintage|neon|noir|black and white|sepia|photo\w*|realistic|render\w*|3d|lego|clay|"
    r"night|day|sunset|sunrise|snow\w*|rain\w*|fog\w*|storm\w*|desert|beach|garage|track|"
    r"red|blue|green|yellow|orange|purple|pink|white|black|gr[ae]y|gold|silver|chrome|matte|"
    r"wheels?|rims?|spoiler|wing|bumper|hood|lights?|smoke|fire|flames?|sticker\w*|plate|"
    r"hat|glasses|sunglasses|helmet|clown|suit|costume|background|sky|colou?rs?|style|"
    r"zoom(?:ed)?|crop(?:ped)?|angle|side|front|rear|top|square|wide)\b",
    re.I,
)
_CONTENT_EDIT_RE = re.compile(r"\b(add|remove|put|swap|replace|change|turn|give it|without|with a|with an|with some|more|less|no)\b", re.I)
_IDIOM_RE = re.compile(
    r"\b(make it (?:stop|quick|snappy|so|happen|rain money|count|work|up)|keep it up|put it this way|"
    r"change the subject|turn it (?:off|down|up)|make it make sense)\b",
    re.I,
)


def looks_like_edit(text: str) -> bool:
    """Do the words ask for a CHANGE to a picture - a visual quality, a thing
    added or removed - rather than comment on one? The classifier said "edit"
    for every caption posted with a picture ("introducing rod 3 to your mothers
    board") and the bot repainted people's photos at them."""
    text = (text or "").strip()
    if not text or _IDIOM_RE.search(text):
        return False
    return bool(_VISUAL_RE.search(text) or _CONTENT_EDIT_RE.search(text))


def parse_followup(text: str) -> tuple[str, str] | None:
    """("again", modification) or ("edit", instruction) when the message is
    about the thing the bot just made, else None. Pure: the caller checks that
    there IS a recent thing, and how recent."""
    text = (text or "").strip()
    if not text or _IDIOM_RE.search(text):
        return None
    m = AGAIN_RE.match(text)
    if m:
        rest = m.group("rest").strip()
        if rest.startswith("?") or text.endswith("?") and not rest.strip("?").strip():
            return None                              # "again?" is a question
        modification = _REST_LEAD_RE.sub("", rest, count=1).strip(" .!?")
        modification = re.sub(r"^(?:again|once more)\b\s*", "", modification, flags=re.I).strip()
        return ("again", modification)
    import imagegen
    if imagegen.wants_edit(text) and (_VISUAL_RE.search(text) or _CONTENT_EDIT_RE.search(text)):
        return ("edit", text)
    return None


# ---- two requests in one message --------------------------------------------

_SPLIT_RE = re.compile(
    r"\s+(?:and then|and also|and|then|plus|also|&)\s+"
    r"(?=(?:also\s+|then\s+)?(?:draw|render|paint|sketch|illustrate|make|write|sing|create|"
    r"gimme|give me|show me|generate|do|compose|drop|spit)\b)",
    re.I,
)
_PRONOUN_OBJECT_RE = re.compile(
    r"\b(about|of|for|on)\s+(?:it|that|this|the same(?: thing| one)?|him|her|them)\b", re.I
)
_PRONOUN_DIRECT_RE = re.compile(
    r"\b(draw|render|paint|sketch|illustrate|sing|rap)\s+(?:it|that|this|him|her|them)\b", re.I
)


def split_requests(text: str) -> list[str]:
    """"draw X and write a song about it" -> ["draw X", "write a song about it"].

    Splits only where a second request VERB follows the joiner, so "pops and
    bangs" and "a gti and a jetta" stay one subject. At most two parts.
    """
    parts = _SPLIT_RE.split((text or "").strip(), maxsplit=1)
    return [p.strip(" ,.") for p in parts if p and p.strip(" ,.")]


def resolve_pronoun(second: str, first_subject: str) -> str:
    """Point the second request's "it/that/him" at what the first was about."""
    subject = (first_subject or "").strip()
    if not subject:
        return second
    out = _PRONOUN_OBJECT_RE.sub(lambda m: f"{m.group(1)} {subject}", second, count=1)
    if out == second:
        out = _PRONOUN_DIRECT_RE.sub(lambda m: f"{m.group(1)} {subject}", second, count=1)
    return out
