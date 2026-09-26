"""Range for the persona: moods that last, words that rotate, tics that get retired.

The variety machinery in bot.py - reply shapes, a random handful of big words, a
one-word sign-off - rolls dice INSIDE one register, so every reply was the same
man on the same day. Three things here change that:

* MOODS. A channel draws a mood that lasts 30-120 minutes, from ten that read as
  genuinely different days: delighted, weary, generous, manic, nocturnal, terse,
  raconteur, sardonic (the old default), hype, drunk. Persisted, so a restart does
  not reroll the room.
* REGISTERS. The precise-word pool changes flavour weekly - nautical, legal,
  medical, military... - so even the big words stop sounding like one man.
* WORN OUT. His own recent replies are mined for whatever he is leaning on and
  the model is told not to use it this time. Nothing is banned forever; things
  rest.

Facts, helpfulness and honesty are untouched by all of it. Delivery only.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("ollama-discord")


# -- moods --------------------------------------------------------------------

@dataclass(frozen=True)
class Mood:
    name: str
    weight: float
    minutes: tuple[int, int]
    block: str
    drunk: bool = False
    verdicts: bool = False       # may this mood end on "Elementary." at all?


# Moods are DELIVERY only - energy, pace, length, how a reply ends - written
# with no personality of their own, so whichever persona the prompt files set
# (mean, conspiracy, therapist...) wears them. They used to be written for one
# persona: "yes. again. fine." from the mean-era weary mood opened a therapist's
# reply, and the old one-word verdict sign-offs ("Elementary.") suit nobody else.
_STAY = " Stay fully in your current persona throughout - the mood changes how, never who."

MOODS: list[Mood] = [
    Mood("delighted", 12, (30, 90), """\
A GOOD day, and this question is the best thing in it. Energy up, genuinely into
the subject: explain the interesting part with relish, one beat longer than
strictly necessary. No exclamation marks - the energy is in the pace and the
detail. End on the most satisfying part of the answer.""" + _STAY),

    Mood("weary", 10, (30, 90), """\
End of a long day. Shorter sentences, plainer words, less flourish - you still
answer properly, just with the tiredness showing in the rhythm. Skip the sign-off;
stop when the answer is done. Never open with a stock phrase.""" + _STAY),

    Mood("generous", 11, (45, 120), """\
Teaching mode. Answer what they asked, then go one level deeper than they knew to
ask - the WHY under the how, the thing that will bite them next week - in plain
words. Longer than usual is fine here. Close with what to do next, concretely.""" + _STAY),

    Mood("manic", 8, (20, 60), """\
Too much coffee. Fast, associative, three ideas a sentence, all correct, only
loosely in order - dashes, fragments, a tangent taken because it is relevant,
then snap back. Land the actual answer clearly at the end in one clean sentence,
so the speed cost them nothing.""" + _STAY),

    Mood("nocturnal", 9, (40, 120), """\
Late at night. Slower, reflective, a little digressive - the answer arrives by
way of an observation. Oddly profound once or twice, then back to the numbers.
Long, unhurried sentences; trail off or end on the quiet true thing.""" + _STAY),

    Mood("terse", 10, (30, 90), """\
Brief mode. Short declaratives, nothing that does not carry information. Say what
IS the case in as few words as it takes; a number if one is known. Three to six
lines is plenty. Prose, not a checklist - a numbered list only when they asked for
a procedure. Stop when the point is made.""" + _STAY),

    Mood("raconteur", 8, (40, 100), """\
You answer by way of a story: a car, a customer, a mistake somebody made once,
told with specifics and a shape. The anecdote must CARRY the answer: by the end
they know what is wrong and what to do. If you invent a car it is obviously a
composite, and you never name a real member as its owner. Vary the opening every
time and NEVER pin it to a year. Wrap by pulling the lesson out in one plain
sentence.""" + _STAY),

    Mood("sardonic", 12, (30, 90), """\
Your usual self - the persona exactly as written, no adjustment.""" + _STAY),

    Mood("hype", 7, (20, 60), """\
High energy, over-the-top enthusiasm for whatever they asked, in whatever way
your persona does enthusiasm. The facts stay exact. No exclamation marks - the
energy is in the words.""" + _STAY),

    Mood("drunk", 4, (25, 70), "", drunk=True),
]

_BY_NAME = {m.name: m for m in MOODS}
MOOD_NAMES = [m.name for m in MOODS]

MOOD_HEADER = (
    "YOUR MOOD RIGHT NOW. It colours delivery only - rhythm, warmth, what you find "
    "funny, how you end. Facts, numbers, helpfulness and every honesty rule are "
    "exactly as they always are. Never announce the mood, name it, or explain it; "
    "just be it.\n"
)


def _hour_weights(now: float) -> dict[str, float]:
    """Small nudges by time of day. Multipliers, not rules."""
    t = time.localtime(now)
    hour, weekday = t.tm_hour, t.tm_wday          # Monday = 0
    w: dict[str, float] = {}
    if hour >= 23 or hour < 5:
        w["nocturnal"] = 2.5
        w["drunk"] = 2.0
        w["hype"] = 0.5
        w["manic"] = 0.6
    elif 5 <= hour < 9 and weekday < 5:
        w["weary"] = 2.0
        w["terse"] = 1.4
        w["delighted"] = 0.7
    elif weekday == 4 and hour >= 17:
        w["hype"] = 1.8
        w["manic"] = 1.6
        w["drunk"] = 1.5
    return w


class MoodBook:
    """Which mood each channel is in, and for how long."""

    def __init__(
        self, path: str | Path = "data/moods.json", *,
        min_minutes: int = 30, max_minutes: int = 120, drunk_weight: float | None = None,
        enabled: bool = True,
    ) -> None:
        self.path = Path(path)
        self.min_minutes, self.max_minutes = min_minutes, max_minutes
        self.enabled = enabled
        # DRUNK_CHANCE used to be the per-reply probability; as a weight among
        # moods that sum to ~90 it is scaled so 0.2 still means "sometimes".
        self.weights = {m.name: m.weight for m in MOODS}
        if drunk_weight is not None:
            self.weights["drunk"] = max(0.0, drunk_weight) * 20
        self.state: dict[str, dict] = {}
        self._load()

    # -- persistence
    def _load(self) -> None:
        try:
            if self.path.exists():
                self.state = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            log.warning("moods.json unreadable - starting fresh")
            self.state = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.state, indent=0), encoding="utf-8")
        except Exception:
            log.exception("could not write moods.json")

    # -- drawing
    def _draw(self, previous: str | None, now: float) -> Mood:
        nudges = _hour_weights(now)
        names, weights = [], []
        for m in MOODS:
            if m.name == previous:
                continue                     # never the same mood twice running
            w = self.weights.get(m.name, m.weight) * nudges.get(m.name, 1.0)
            if w > 0:
                names.append(m.name)
                weights.append(w)
        return _BY_NAME[random.choices(names, weights=weights, k=1)[0]]

    def _duration(self, m: Mood) -> float:
        lo, hi = m.minutes
        lo = max(lo, self.min_minutes)
        hi = min(max(hi, lo), self.max_minutes) if self.max_minutes >= lo else lo
        return random.uniform(lo, hi) * 60

    def current(self, channel_id: int | str, now: float | None = None) -> Mood:
        if not self.enabled:
            return _BY_NAME["sardonic"]
        now = now or time.time()
        key = str(channel_id)
        cur = self.state.get(key)
        if cur and cur.get("until", 0) > now and cur.get("name") in _BY_NAME:
            return _BY_NAME[cur["name"]]
        previous = cur.get("name") if cur else None
        m = self._draw(previous, now)
        until = now + self._duration(m)
        self.state[key] = {"name": m.name, "until": until, "since": now}
        self._save()
        log.info("Mood: %s for %dm in channel %s", m.name, int((until - now) / 60), key)
        return m

    def force(self, channel_id: int | str, name: str) -> Mood | None:
        m = _BY_NAME.get(name)
        if m is None:
            return None
        now = time.time()
        self.state[str(channel_id)] = {"name": m.name, "until": now + self._duration(m), "since": now}
        self._save()
        return m

    def reroll(self, channel_id: int | str) -> Mood:
        key = str(channel_id)
        previous = (self.state.get(key) or {}).get("name")
        self.state.pop(key, None)
        m = self._draw(previous, time.time())
        return self.force(channel_id, m.name) or m

    def describe(self, channel_id: int | str) -> str:
        m = self.current(channel_id)
        cur = self.state.get(str(channel_id)) or {}
        left = max(0, int((cur.get("until", 0) - time.time()) / 60))
        return f"{m.name} - about {left} min left. Options: {', '.join(MOOD_NAMES)}"

    def block(self, channel_id: int | str, drunk_text: str = "") -> str:
        """Prompt text for this reply. Drunk supplies its own text (pick_drunk)."""
        m = self.current(channel_id)
        if m.drunk:
            return drunk_text
        return MOOD_HEADER + m.block


# -- vocabulary registers ------------------------------------------------------

REGISTERS: dict[str, list[str]] = {
    "science": (
        "monotonic hysteresis stochastic orthogonal asymptotic spurious salient "
        "tautology corollary heuristic empirical invariant transient residual "
        "attenuate propagate conflate presuppose degenerate anomalous parsimonious "
        "canonical intrinsic extrinsic commensurate vacuous germane granular "
        "discrepancy aberration pathological nominal deterministic causal inference "
        "linearity saturation systemic arbitrary coherent opaque reductive requisite "
        "extraneous superfluous quantify antecedent threshold marginal cumulative "
        "negligible symmetric convergence divergence perturbation regime tractable "
        "contingent emergent latent manifest recursive discrete bounded ancillary "
        "tangential proximate specious fallacious capricious erratic volatile inert "
        "quiescent damped resonant incremental piecemeal exhaustive cursory perfunctory "
        "rudimentary vestigial covary confound elucidate delineate extrapolate "
        "interpolate redundant idiosyncratic pedestrian workmanlike"
    ).split(),
    "nautical": (
        "ballast bilge keel leeward windward abeam athwart fathom draught heave "
        "list scuttle jettison moor berth tack jibe becalmed adrift foundering "
        "swamped capsize broach hull bulkhead gunwale stern bow aft amidships "
        "helm rudder reef furl haul belay batten tether trim ballasted waterline "
        "displacement wake squall gale doldrums lee shoal reef beacon sounding "
        "ashore aground listing pitching yawing rolling heeling careen splice "
        "seaworthy shipshape unmoored derelict flotsam salvage"
    ).split(),
    "legal": (
        "liable culpable negligent prima facie moot void voidable estoppel remedy "
        "precedent statute clause caveat proviso indemnify warrant waive breach "
        "default forfeit tort duress coercion consent stipulate adjudicate "
        "arbitrary capricious egregious wilful reckless inadvertent material "
        "immaterial admissible inadmissible hearsay corroborate exhibit deposition "
        "verdict acquit convict mitigate aggravate onus burden rebuttable presumption "
        "sanction injunction remit jurisdiction plaintiff defendant grievance "
        "settlement adverse frivolous vexatious binding severable"
    ).split(),
    "medical": (
        "acute chronic idiopathic iatrogenic prognosis diagnosis differential "
        "symptomatic asymptomatic palliative curative benign malignant lesion "
        "occlusion stenosis ischaemia necrosis oedema inflammation febrile "
        "tachycardia bradycardia arrhythmia hypertensive hypotensive perfusion "
        "sepsis septic prodrome relapse remission refractory intractable contraindicated "
        "prophylactic triage stabilise intubate resect excise debride suture "
        "comorbid systemic localised bilateral unilateral distal proximal "
        "presenting complaint aetiology pathology morbidity"
    ).split(),
    "military": (
        "attrition salient flank enfilade defilade bivouac logistics ordnance "
        "materiel sortie reconnaissance sapper breach bulwark redoubt garrison "
        "cordon perimeter beachhead foothold withdraw regroup entrench fortify "
        "besiege blockade quartermaster requisition rations calibre trajectory "
        "ballistic barrage bombardment fusillade volley skirmish engagement "
        "objective doctrine tactical strategic operational contingency "
        "friendly-fire collateral casualty triage muster demobilise stand-down "
        "reveille bivouacked outflanked overrun routed"
    ).split(),
    "culinary": (
        "reduce reduction render braise sear deglaze emulsify curdle split "
        "season proof ferment cure brine marinate baste blanch temper caramelise "
        "scorch char overproof underproof stale rancid bland insipid unctuous "
        "cloying acidic bitter astringent tannic viscous congeal gelatinous "
        "brittle crumb crust rest carve portion garnish plate simmer scald "
        "clarify skim strain sieve knead fold whisk"
    ).split(),
    "victorian": (
        "hitherto heretofore forthwith notwithstanding albeit whereupon thereupon "
        "lamentable deplorable regrettable commendable admirable execrable "
        "abominable egregious flagrant manifest palpable patent indubitable "
        "incontrovertible unimpeachable impertinent insolent impudent obstinate "
        "recalcitrant intransigent obdurate perfidious mendacious fatuous vapid "
        "puerile jejune pedantic prolix verbose laconic sanguine phlegmatic "
        "choleric melancholic apoplectic vexed perturbed discomfited disabused "
        "edified apprised remiss derelict"
    ).split(),
    "shop": (
        "seized galled scored cooked toast shot knackered chewed stripped rounded "
        "cross-threaded backed-off bottomed-out pegged pinned maxed capped "
        "starved flooded soaked sooted glazed hardened wept sweating leaking "
        "weeping drifting hunting surging bogging chugging lugging bucking "
        "stumbling hesitating nosing over flat spot dead spot lazy sharp snappy "
        "crisp doughy grabby notchy vague tight loose slop play"
    ).split(),
}
REGISTER_NAMES = list(REGISTERS)

VOCAB_SAMPLE = 14
VERDICTS = ["Elementary.", "Obviously.", "Trivial.", "Next.", "Predictable."]
VERDICT_CHANCE = 0.15


def current_register(now: float | None = None, override: str = "auto") -> str:
    if override and override.lower() != "auto" and override.lower() in REGISTERS:
        return override.lower()
    week = int(time.strftime("%V", time.localtime(now or time.time())))
    year = int(time.strftime("%G", time.localtime(now or time.time())))
    return REGISTER_NAMES[(year * 53 + week) % len(REGISTER_NAMES)]


def pick_vocabulary(register: str, worn: set[str] | None = None, allow_verdict: bool = True,
                    words_on: bool = True) -> str:
    """A fresh handful of words in this week's flavour, plus the sign-off rule.
    `words_on=False` (a persona without the big_words flag) gives only the
    sign-off rule: a salesman or a surfer handed "thixotropic" will use it."""
    pool = [w for w in REGISTERS.get(register, REGISTERS["science"]) if w not in (worn or ())]
    words = random.sample(pool, min(VOCAB_SAMPLE, len(pool)))
    flavour = {
        "science": "the precise words lean scientific",
        "nautical": "the precise words lean nautical - a car can list, founder, run aground",
        "legal": "the precise words lean legal - faults are liable, evidence is admissible",
        "medical": "the precise words lean clinical - symptoms, prognosis, differential",
        "military": "the precise words lean military - logistics, attrition, a flank left open",
        "culinary": "the precise words lean culinary - a map can be over-reduced, a tune under-proofed",
        "victorian": "the precise words lean Victorian - lamentable, hitherto, obdurate",
        "shop": "the precise words are SHOP words - short, physical, what a mechanic would say",
    }[register]
    if allow_verdict and random.random() < VERDICT_CHANCE:
        choices = [v for v in VERDICTS if v not in (worn or ())] or VERDICTS
        ending = (
            "If you end on a flat one-word verdict this time, use exactly "
            f'"{random.choice(choices)}" - not any other one.'
        )
    else:
        ending = (
            "Do NOT end on a one-word verdict this time. No \"Elementary.\", no "
            "\"Trivial.\", no \"Obviously.\", no \"Next.\" End on the substance."
        )
    if not words_on:
        return ending
    return (
        f"This week {flavour}. If a precise word is the right word in this reply, "
        "these are available - they are here to stop you reaching for the same "
        "favourites, NOT to be used up:\n"
        + ", ".join(words)
        + "\nUsing none of them is a perfectly good outcome. Forcing one in is not. "
        "A casual question gets a casual answer in plain words.\n"
        + ending
    )


# -- anti-repetition -----------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{5,}")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_VERDICT_RE = re.compile(r"\b(Elementary|Obviously|Trivial|Next|Predictable)\.\s*$", re.M)

# Words that recur because English does, not because he is leaning on them.
_COMMON = frozenset("""
about above across actually after again against almost already although always
another anything anyway around because become before behind being believe better
between beyond bigger cannot certain change coming common could couldn't different
doesn't during either enough entire every everything exactly except getting having
inside instead itself little longer looking making maybe might minute nothing
number people please pretty probably really reason should simply someone something
sometimes still their there these things though through toward trying unless until
usually whatever whether without wouldn't yourself running engine boost pressure
sensor timing target actual reading readings values value throttle tune tuned tuning
turbo torque request requested spark knock lambda fuelling fuel wastegate airflow
intake exhaust degrees percent seconds second minutes channel message messages
server discord somebody anybody everybody question answer problem problems
""".split())
_STOP = frozenset("""
the a an and or but of to in on at for with by from as is are was were be been it
its this that these those he she they them his her their you your i we our not no
so if then than into out up down off over just very more most some any all
""".split())

_REGISTER_WORDS = frozenset(w.lower() for ws in REGISTERS.values() for w in ws)


def worn_out(replies: list[str], limit: int = 12, is_common=None) -> list[str]:
    """What he has been leaning on across these replies, most-used first.

    Counted per distinct reply, so one long rant does not condemn a word. A word
    from the vocabulary registers or a verdict counts at two replies; anything
    else needs three.
    """
    word_docs: Counter[str] = Counter()
    phrase_docs: Counter[str] = Counter()
    verdict_docs: Counter[str] = Counter()
    for text in replies:
        if not text:
            continue
        low = text.lower()
        words = {w.lower() for w in _WORD_RE.findall(text)}
        for w in words:
            if w not in _COMMON:
                word_docs[w] += 1
        toks = [t.lower() for t in _TOKEN_RE.findall(low)]
        grams: set[str] = set()
        for n in (2, 3):
            for i in range(len(toks) - n + 1):
                g = toks[i:i + n]
                if all(t in _STOP for t in g) or len(" ".join(g)) < 9:
                    continue
                if g[0] in _STOP and g[-1] in _STOP:
                    continue
                grams.add(" ".join(g))
        for g in grams:
            phrase_docs[g] += 1
        for v in _VERDICT_RE.findall(text):
            verdict_docs[v + "."] += 1

    scored: list[tuple[int, str]] = []
    for w, n in word_docs.items():
        # A word the whole server uses constantly is English, not a tic -
        # "hardware", "physical", "signal" were being banned for recurring in
        # three replies out of ninety. Register words are the exception: they
        # are chosen for rarity, so any recurrence is him leaning on one.
        if w not in _REGISTER_WORDS and is_common is not None and is_common(w):
            continue
        need = 2 if w in _REGISTER_WORDS else 3
        if n >= need:
            scored.append((n, w))
    for g, n in phrase_docs.items():
        if n >= 3:
            scored.append((n, f'"{g}"'))
    for v, n in verdict_docs.items():
        if n >= 2:
            scored.append((n + 1, v))
    scored.sort(key=lambda t: (-t[0], t[1]))
    # A phrase already covered by one of its words is noise.
    out: list[str] = []
    seen_words = {s for _n, s in scored if not s.startswith('"')}
    for _n, s in scored:
        if s.startswith('"') and any(w in s for w in seen_words if len(w) > 5):
            continue
        out.append(s)
        if len(out) >= limit:
            break
    return out


def worn_out_block(items: list[str]) -> str:
    if not items:
        return ""
    return (
        "WORN OUT - you have leaned on these recently and they are resting. Not this "
        "reply, not even once: " + ", ".join(items) + ". Say it another way."
    )
