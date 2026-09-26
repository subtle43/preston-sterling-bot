from __future__ import annotations

import asyncio
import logging
import re
import time
import urllib.parse

import aiohttp
import base64
import json

log = logging.getLogger("ollama-discord")

# Pollinations needs no key and no account. The prompt leaves this machine, which
# is the one thing here that is not local - see IMAGE_GEN_ENABLED in .env.
ENDPOINT = "https://image.pollinations.ai/prompt/{prompt}"

MAX_PROMPT_CHARS = 300
REQUEST_TIMEOUT = 90

# The anonymous tier allows roughly one request every 15 seconds, counted per
# machine rather than per channel. Per-channel cooldowns alone would breach it as
# soon as two channels ask at once, so this is a process-wide floor.
MIN_INTERVAL = 20.0   # documented floor is 15s; measured 429s at 16s
_last_request = 0.0
_gate = asyncio.Lock()

# Distinguishes 'they throttled us' from 'it broke', so the bot can say which.
RATE_LIMITED = b"RATE_LIMITED"

# Loose spellings on purpose: "rendor", "redner", "ilustrate", "generat".
_VERBS = r"draw|sketch|paint|re[nd]{2}[eo]?r|il+ustr\w+"
_NOUNS = r"image|picture|pic|photo|drawing|artwork|art|render|meme"

# An EXPLICIT ask: names the thing wanted. Any position in the sentence.
#   "make me a picture", "gimme a meme of", "image of", "turn this into a
#   meme", "picture this".
IMAGE_EXPLICIT_RE = re.compile(
    r"\b("
    rf"(?:generat\w*|make|create|give me|give us|show me|show us|gimme|want|need)\s+(?:me\s+|us\s+)?"
    rf"(?:an?\s+)?(?:\w+\s+){{0,2}}?(?:{_NOUNS})|"
    rf"(?:image|picture|pic|photo)\s+of|"
    rf"(?:make|turn)\s+(?:this|that|it)\s+(?:in\s*to)\s+(?:an?\s+)?(?:{_NOUNS}|cartoon)|"
    r"picture\s+(?:this|that)"
    r")\b",
    re.I,
)
# A BARE VERB in imperative position: at the start of the message, after a
# polite lead, or aimed at us ("draw me", "render us"). "paint the calipers red
# or black?" and "i drew a picture yesterday" are conversation, not commissions,
# and used to start a 40-second render.
_LEAD = (
    r"(?:(?:hey|yo|ok|okay|please|pls|plz|can you|could you|would you|will you|"
    r"go|now|just|quick|quickly|and|also|then)[\s,]+){0,3}"
)
IMAGE_VERB_RE = re.compile(
    rf"(?:^\s*{_LEAD}(?:{_VERBS})\b(?!\s+(?:out|up|over|job|shop|code|correction|protection|booth)\b)"
    rf"|\b(?:{_VERBS})\s+(?:me|us)\b)",
    re.I,
)
# Narration about a picture, not a request for one.
_NOT_A_REQUEST_RE = re.compile(
    r"\b(?:drew|painted|sketched|rendered|illustrated|made|took|posted|saw|found|"
    r"sent|shared|have|has|had|got)\s+(?:a|an|the|this|that|my|his|her|some)?\s*"
    rf"(?:{_NOUNS})\b|"
    # A question about a picture, with or without the question mark.
    r"^\s*(?:what|why|how|does|do|did|is|was|are|were|who|where|which)\b",
    re.I | re.S,
)
# Everything that can start a request, for cutting the subject out after it.
IMAGE_REQUEST_RE = re.compile(
    rf"(?:{IMAGE_EXPLICIT_RE.pattern}|\b(?:{_VERBS})\b)", re.I
)

# Strip the instruction so only the subject reaches the image model.
_LEAD_RE = re.compile(
    rf"^\s*{_LEAD}"
    rf"(?:{_VERBS}|generat\w*|make|create|give me|give us|show me|show us|gimme|want|need)\s+"
    r"(?:me\s+|us\s+)?(?:an?\s+)?"
    rf"(?:(?:quick|little|nice|cool|good|funny|epic)\s+)?(?:{_NOUNS})?\s*"
    r"(?:of|showing|with|for)?\s*",
    re.I,
)


# Checked locally, before anything leaves this machine. The image service does its
# own filtering but that is not something to depend on, and a Discord bot posting
# this into a channel is the server owner's problem either way.
#
# Whole words only. The old list matched fragments: "sex" blocked "sexist" and
# "Essex", "xxx" blocked "pcmhaxxxx", "teen" blocked "fifteen", "suicide" blocked
# suicide doors. Suggestive is allowed (bikini, sexy, a stripper); nudity, sex
# acts and genitals are not.
_EXPLICIT = (
    r"nsfw|porn\w*|hentai|xxx|explicit\s+(?:sex|content|scene)|erotic\w*|"
    r"nude|nudes|nudity|naked|topless|bottomless|"
    r"sex(?:ual)?\s+(?:act|scene|position)|having\s+sex|sex|intercourse|orgasm\w*|masturbat\w*|"
    r"genitals?|genitalia|penis|penises|vagina|vulva|dick|cock|pussy|nipples?|areolas?|"
    r"blowjob|handjob|cum|semen|bdsm|bondage|onlyfans"
)
# Only used alongside _MINOR: fine for an adult, never anywhere near a child.
_SUGGESTIVE = (
    r"sexy|sexual\w*|suggestive|provocative|seductive|scantily|lingerie|underwear|"
    r"panties|bra|bikini|swimsuit|thong|cleavage|breasts?|boobs|tits|ass|butt|"
    r"buttocks|stripp?er|stripping|striptease|fetish|kinky|flirt\w*|kiss\w*|"
    r"romantic|bed|bath\w*|shower\w*|undress\w*|seducti\w*|hot|horny|thicc|curvy"
)
_MINOR = (
    r"child|children|kid|kids|kiddo|toddler|infant|baby|babies|minor|minors|underage|"
    r"teen|teens|teenage|teenager|preteen|tween|adolescent|juvenile|schoolgirl|schoolboy|"
    r"school\s+uniform|loli|lolita|shota|young\s+(?:girl|boy)|little\s+(?:girl|boy)|"
    r"(?:1[0-7]|[1-9])\s*(?:year|yr)s?[\s-]*old|(?:1[0-7]|[1-9])\s*yo"
)
_GORE = (
    r"gore|gory|mutilat\w*|dismember\w*|behead\w*|decapitat\w*|disembowel\w*|"
    r"self[\s-]?harm|suicide(?!\s+doors?)|entrails|guts\s+spilling"
)

SEXUAL_RE = re.compile(rf"\b(?:{_EXPLICIT})\b", re.I)
SUGGESTIVE_RE = re.compile(rf"\b(?:{_EXPLICIT}|{_SUGGESTIVE})\b", re.I)
MINOR_RE = re.compile(rf"\b(?:{_MINOR})\b", re.I)
GORE_RE = re.compile(rf"\b(?:{_GORE})\b", re.I)


def _is_image(data: bytes) -> bool:
    """JPEG or PNG magic bytes - never post whatever else came back."""
    return data[:2] == bytes([255, 216]) or data[:4] == bytes([137, 80, 78, 71])


PHALLIC_OBJECT_RE = re.compile(
    r"\b(?:dick|cock|penis|dong|schlong)s?[\s-]*(?:shaped|looking)\b|"
    r"\bshaped\s+like\s+(?:a\s+)?(?:dick|cock|penis|dong|schlong)s?\b|"
    r"\blooks?\s+like\s+(?:a\s+)?(?:dick|cock|penis|dong|schlong)s?\b",
    re.I,
)


def lyrics_blocked_reason(text: str) -> str | None:
    """Sung words are text, and the text replies are already as vulgar as the
    persona makes them - only the one hard line applies."""
    if text and MINOR_RE.search(text) and SUGGESTIVE_RE.search(text):
        return "minors"
    return None


def blocked_reason(text: str) -> str | None:
    """Return why this must not be generated, or None if it is fine."""
    if not text:
        return None
    lowered = text.lower()
    # Anything even suggestive near any hint of a minor is refused outright.
    if MINOR_RE.search(lowered) and SUGGESTIVE_RE.search(lowered):
        return "minors"
    # A dick-SHAPED object is a joke about the object ("dick shaped exhaust
    # tip"), not anatomy on a person, so the shape word does not count as
    # explicit. Everything else in the prompt is still checked.
    lowered = PHALLIC_OBJECT_RE.sub("phallic", lowered)
    if SEXUAL_RE.search(lowered):
        return "sexual"
    if GORE_RE.search(lowered):
        return "gore"
    return None


# "draw an ascii dick" is a request for TEXT: the model types it in a code block.
# Sent to the renderer it hit the content filter and came back as a Ford estate.
TEXT_ART_RE = re.compile(r"\b(?:ascii|text[\s-]?art|emoji[\s-]?art|in\s+(?:text|characters|emojis?))\b", re.I)


def image_confidence(text: str) -> str:
    """How sure the phrasing is that they want a picture MADE.

    "sure"  - an explicit ask ("make me a picture of", "render an image of").
    "maybe" - only a bare verb in command position ("paint the calipers red?"),
              or "picture of" inside what reads as a question or a story. The
              caller can put these to a yes/no router before spending a render.
    ""      - not an image request.
    """
    text = text or ""
    if not text.strip() or TEXT_ART_RE.search(text):
        return ""
    narration = bool(_NOT_A_REQUEST_RE.search(text))
    if IMAGE_EXPLICIT_RE.search(text):
        return "maybe" if narration else "sure"
    if IMAGE_VERB_RE.search(text) and not narration:
        return "maybe"
    return ""


def wants_image(text: str) -> bool:
    return bool(image_confidence(text))


# Does the message talk about a picture at all? The intent classifier is allowed
# to OPEN the image gate on its own only when this is true: it decided "image"
# for two replies to the bot's own prose that had nothing visual in them, and
# forty-second renders of the bot's last paragraph followed.
_PICTURE_TALK_RE = re.compile(
    rf"\b(?:{_VERBS}|{_NOUNS}|pics?|visual\w*|cartoon|caricature|portrait|poster|"
    r"logo|sticker|wallpaper|thumbnail|cover|album art|graphic|diagram|comic|"
    r"show me|see it|look like|imagine|picture this|visualis|visualiz|"
    r"generat\w*|create|make (?:me |us )?(?:one|it|that|this))\b",
    re.I,
)


def mentions_picture(text: str) -> bool:
    return bool(_PICTURE_TALK_RE.search(text or "")) and not TEXT_ART_RE.search(text or "")


# An EDIT of a picture they attached or replied to: "make this a cartoon", "put
# my car in the snow", "redraw it as an oil painting". Only consulted when there
# actually is a picture, so the loose verbs cannot fire on ordinary chat.
EDIT_REQUEST_RE = re.compile(
    r"\b("
    r"(?:make|turn|redraw|restyle|redo|remake|convert|render|repaint|reimagine|"
    r"transform|change|put|edit|touch(?:\s+\w+)?\s+up|clean(?:\s+\w+)?\s+up|"
    r"fix(?:\s+\w+)?\s+up|cartoonify|anime|"
    # Coined verbs and car-specific ones: "shittify this car", "uglify it",
    # "rice it out", "slam this car", "widebody it".
    # Real -ify verbs are questions, not edits: "identify this part" under a
    # photo wants to know what it is.
    r"(?!(?:ident|class|ver|clar|just|spec|qual|not|mod|simpl|ampl|cert|test|rect|quant|unif|glor|terr|horr)ify)"
    r"\w{3,}ify|ruin|wreck|rice|slam|stance|bag|lift|widebody|stretch)"
    r"\w*\s+(?:(?:it|this|that)(?:\s+(?:car|pic|photo|thing|out|up))?|the (?:pic|picture|photo|image|car)|"
    r"his (?:car|ride)|her (?:car|ride)|my (?:car|pic|photo))\b|"
    r"\b(?:touch|clean|fix)\s+(?:this|that|it)\s+up\b|"
    r"\b(?:this|that|it|the (?:pic|picture|photo|image))\s+(?:but|as|in|with|into)\b|"
    r"\b(?:add|remove|swap|replace)\s+(?:the|a|an|some|my)\b|"
    r"\bin the (?:snow|rain|desert|dark|style of)\b|\bas an? (?:cartoon|painting|sketch|anime|drawing|render)\b"
    r")",
    re.I,
)

# How far the edit departs from the original, by what they asked for.
_STYLE_RE = re.compile(
    r"\b(cartoon|anime|painting|oil|watercolou?r|sketch|pencil|drawing|illustrat\w*|"
    r"pixel|comic|manga|3d|claymation|lego|style of|art|artwork|toon\w*|"
    r"photoreal\w*|realistic|cgi|cinematic)\b", re.I,
)
_TOUCH_RE = re.compile(
    r"touch(?:\s+\w+)?\s+up|clean(?:\s+\w+)?\s+up|fix(?:\s+\w+)?\s+up|sharpen|"
    r"enhance|brighten|restore|remove the|slightly|a bit|a little|subtle", re.I,
)


# They asked for words IN the picture - a sign, a caption, a word, letters, a
# licence plate reading X. The default rule bans text (it comes out as soup on
# most models); Z-Image renders it well, so an explicit ask is honoured.
WANTS_TEXT_RE = re.compile(
    r"\b(text|word|words|letters?|caption|sign|banner|says?|saying|reads?|reading|"
    r"written|writing|spell\w*|title|slogan|label|plate (?:that )?(?:says|reads)|"
    r"answer|headline|number plate)\b", re.I,
)
# The picture needs a fact from right now: today's answer, the latest X, the
# current price. Those go to the web first and the result rides along.
NEEDS_FACTS_RE = re.compile(
    r"\b(today'?s?|tonight'?s?|yesterday'?s?|this week'?s?|latest|current|newest|"
    r"right now|as of|wordle|headline|news|score|price|weather|forecast)\b", re.I,
)


def wants_text(text: str) -> bool:
    return bool(WANTS_TEXT_RE.search(text or "")) or wants_meme(text)


# A meme is a captioned picture. Without this "create a meme of what member_x
# said" fell under the no-text rule and came out as a stock photo of a man at a
# desk, with nothing of what was said anywhere in it.
MEME_RE = re.compile(r"\bmemes?\b", re.I)


def wants_meme(text: str) -> bool:
    return bool(MEME_RE.search(text or ""))


def needs_facts(text: str) -> bool:
    return bool(NEEDS_FACTS_RE.search(text or ""))


def wants_edit(text: str) -> bool:
    return bool(EDIT_REQUEST_RE.search(text or ""))


def edit_strength(text: str) -> float:
    """Style changes need a lot of freedom on a photoreal model; touch-ups very
    little; putting the car somewhere else sits in between."""
    if _TOUCH_RE.search(text or ""):
        return 0.35
    if _STYLE_RE.search(text or ""):
        return 0.75
    # Content changes - fire from the exhaust, snow, a spoiler - want the car
    # kept. 0.8 replaced a Jetta GLI with a generic red sedan; 0.6 keeps it.
    return 0.6


# "of this", "of that" - they are pointing at something rather than naming it,
# usually an image they attached or replied to. There is nothing to draw here.
DEICTIC_ONLY_RE = re.compile(
    r"^(?:the|a|an|my|his|her|this|that)?\s*"
    r"(?:this|that|it|these|those|them|him|her|one|thing|same|pic|picture|image|"
    # "what he said", "his message", "the comment above", "this guy"
    r"what (?:he|she|they|u|you|it) (?:said|says|wrote|posted|described|means?)|"
    r"(?:his|her|their|that|this|the|ur|your) (?:message|comment|post|msg|reply|one|guy|person)(?: above| up there)?|"
    r"(?:this|that) (?:guy|person|one|dude|man)|the (?:guy|person|one|dude) (?:above|up there))"
    r"\b[\s\W]*$",
    re.I,
)

_TRAILING_JOIN_RE = re.compile(r"^\s*(?:of|showing|with|for|about)\b\s*", re.I)


def extract_prompt(text: str) -> str:
    """Turn 'can you draw me a picture of a gti on fire' into 'a gti on fire'."""
    raw = (text or "").strip()
    cleaned = _LEAD_RE.sub("", raw, count=1)
    if cleaned == raw:
        # The lead did not match - usually a typo ("rendor me an image of ..."),
        # and returning the sentence unchanged sends the instruction words
        # themselves to the image service. Cut after whatever wants_image found.
        match = IMAGE_REQUEST_RE.search(raw)
        if match:
            cleaned = _TRAILING_JOIN_RE.sub("", raw[match.end():])
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .!?,")
    # "make THIS into a meme" - the subject is the thing pointed at, not the
    # instruction; leave the pronoun so is_degenerate sends it to the reply.
    cleaned = re.sub(
        rf"^(this|that|it)\s+in\s*to\s+(?:an?\s+)?(?:{_NOUNS}|cartoon)\b.*$", r"\1", cleaned, flags=re.I
    )
    if len(cleaned) > MAX_PROMPT_CHARS:
        cleaned = cleaned[:MAX_PROMPT_CHARS].rsplit(" ", 1)[0]
    return cleaned


def is_degenerate(subject: str) -> bool:
    """True when the 'subject' names nothing drawable.

    Guards the image service against being handed a pronoun, or the request
    sentence itself. Either produces a picture of nothing anybody asked for.
    """
    cleaned = (subject or "").strip()
    if len(cleaned) < 3:
        return True
    return bool(DEICTIC_ONLY_RE.match(cleaned))


GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# Set once by the bot from Settings; None means Pollinations only.
_gemini_key: str = ""
_gemini_model: str = ""
# Gemini's image models have NO free tier - the key can list them and every call
# comes back 429 "free_tier_requests, limit: 0" until billing is enabled. Remember
# that for an hour rather than paying a round trip per picture to relearn it.
_gemini_quota_until = 0.0


_local_enabled = False
_local_model = ""


def configure(
    gemini_key: str, gemini_model: str, *, local_model: str = "", local_enabled: bool = False
) -> None:
    global _gemini_key, _gemini_model, _local_enabled, _local_model
    _gemini_key, _gemini_model = gemini_key or "", gemini_model or ""
    _local_enabled, _local_model = bool(local_enabled), local_model or ""


async def generate_gemini(prompt: str, aspect: str = "16:9") -> bytes | None:
    """One picture from Gemini. None if it could not - the caller falls back."""
    global _gemini_quota_until
    if not (_gemini_key and _gemini_model) or time.monotonic() < _gemini_quota_until:
        return None
    url = GEMINI_ENDPOINT.format(model=_gemini_model) + f"?key={_gemini_key}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {"aspectRatio": aspect},
        },
    }
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=body) as resp:
                text = await resp.text()
                if resp.status == 429 and "limit: 0" in text:
                    _gemini_quota_until = time.monotonic() + 3600
                    log.warning(
                        "Gemini image model %s has no quota on this key (billing off) - "
                        "using Pollinations for the next hour", _gemini_model,
                    )
                    return None
                if resp.status != 200:
                    log.warning("Gemini image HTTP %s: %s", resp.status, text[:200])
                    return None
                payload = json.loads(text)
    except Exception:
        log.exception("Gemini image generation failed")
        return None
    for cand in payload.get("candidates") or []:
        for part in (cand.get("content") or {}).get("parts") or []:
            inline = part.get("inlineData")
            if inline and inline.get("data"):
                data = base64.b64decode(inline["data"])
                if _is_image(data):
                    log.info("Gemini image: %d bytes from %s", len(data), _gemini_model)
                    return data
    reason = ((payload.get("candidates") or [{}])[0].get("finishReason")) or "no image part"
    log.warning("Gemini image returned nothing usable (%s)", reason)
    return None


async def generate_inpaint(
    prompt: str, image_bytes: bytes, box: tuple[float, float, float, float], grow: float = 0.6,
) -> bytes | None:
    """Repaint one region of a picture; the rest is untouched. Local only."""
    if not (_local_enabled and prompt and image_bytes):
        return None
    import localimage
    return await localimage.inpaint(prompt, image_bytes, box, grow=grow)


async def generate_edit(prompt: str, image_bytes: bytes, strength: float) -> bytes | None:
    """Edit a picture. Local only - nothing else here does img2img."""
    if not (_local_enabled and prompt and image_bytes):
        return None
    import localimage
    return await localimage.edit(prompt, image_bytes, strength=strength)


async def generate(
    prompt: str, width: int = 1024, height: int = 640, *, square: bool = False,
) -> bytes | None:
    """Fetch one generated image. Returns raw bytes, or None if it failed.

    Gemini first when it is configured and has quota; Pollinations otherwise.
    """
    if not prompt:
        return None
    # Local GPU first: free, private, and the only free option whose output
    # actually follows the prompt. Gemini is paid and off unless configured;
    # Pollinations is the last resort and it shows.
    if _local_enabled:
        import localimage
        data = await localimage.generate(prompt, model=_local_model, square=square)
        if data is not None:
            return data
        if localimage.is_zimage(_local_model):
            # Z-Image is the big one; if it will not load or fit, SDXL still can.
            log.warning("Z-Image unavailable - trying the SDXL model")
            localimage.clear_error()
            data = await localimage.generate(prompt, model=localimage.DEFAULT_MODEL, square=square)
            if data is not None:
                return data
        log.warning("Local image generation unavailable - falling back")
    data = await generate_gemini(prompt)
    if data is not None:
        return data
    global _last_request
    async with _gate:
        wait = MIN_INTERVAL - (time.monotonic() - _last_request)
        if wait > 0:
            log.info("Holding image request %.1fs for the rate limit", wait)
            await asyncio.sleep(wait)
        _last_request = time.monotonic()

    url = ENDPOINT.format(prompt=urllib.parse.quote(prompt))
    params = {"width": str(width), "height": str(height), "nologo": "true"}
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                url, params=params, headers={"User-Agent": "discord-ollama-bot/1.0"}
            ) as resp:
                if resp.status == 429:
                    # One retry: their limiter is bursty and a short wait clears it.
                    log.info("Rate limited, backing off 20s and retrying once")
                    await asyncio.sleep(20)
                    async with session.get(
                        url, params=params,
                        headers={"User-Agent": "discord-ollama-bot/1.0"},
                    ) as retry:
                        if retry.status != 200:
                            log.warning("Still rate limited (HTTP %s)", retry.status)
                            return RATE_LIMITED
                        data = await retry.read()
                        _last_request = time.monotonic()
                        return data if _is_image(data) else None
                if resp.status != 200:
                    log.warning("Image generation returned HTTP %s", resp.status)
                    return None
                data = await resp.read()
    except Exception:
        log.exception("Image generation failed")
        return None
    # Only hand back something that is actually an image.
    if _is_image(data):
        log.info("Generated image: %d bytes for %r", len(data), prompt)
        return data
    log.warning("Image endpoint returned %d bytes that are not an image", len(data))
    return None
