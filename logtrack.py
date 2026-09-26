"""Read a datalog the way a tuner does: across the pull, not as aggregates.

min/max/mean per channel answers almost nothing useful. "Peak boost 36.1" does not
say whether boost held or fell over; "lambda min 0.77" does not say whether the
mixture tracked its command. Both are questions about how a value behaved ACROSS
the run, and a language model cannot answer them by scanning rows - it
approximates, and approximating a datalog is how somebody melts a piston.

So the arithmetic happens here. Every closed loop in the log is paired with its own
setpoint automatically, binned by engine speed, and handed over as a small table.
"""

from __future__ import annotations

import re

# "Lambda SP", "MAP SP", "Torque Req", "Airmass Soll" - the commanded half of a pair.
SETPOINT_RE = re.compile(
    r"\b(sp|set.?point|target|soll|req|reqd|cmd|desired|dmd|dsr)\b", re.I
)
RPM_RE = re.compile(r"engine.?speed|drehzahl|\bneng\b|\brpm\b", re.I)
BOOST_RE = re.compile(r"boost|\bmap\b|manifold|\bput\b|psig|\bpsi\b|ladedruck", re.I)
KNOCK_RE = re.compile(r"knock|klopf|retard|k.?ret", re.I)
TIMING_RE = re.compile(r"ign.*tim|timing|spark|\biga\b|\bzw\b", re.I)

# Which loops get a column of their own. The rest are still paired and summarised.
HEADLINE = ("lambda", "boost", "map", "put", "airmass", "fuel flow", "torque")

# Loops whose setpoint is routinely overridden on a tuned car, so actual missing
# commanded is the tune working, not a fault. These get their numbers reported
# and no verdict - flagging torque request vs actual led a review with a finding
# that was intentional, on a car deliberately tuned to PUT SP instead.
#
# Airmass is the same story and for the same reason: the airmass setpoint is
# deliberately left high and boost is limited by PUT SP instead, so airmass
# sitting under its command is the tune doing exactly what it was asked. It was
# leading reviews as the headline deficit (-37% at 4000 rpm on a healthy pull)
# when nobody tunes airmass to hit target in the first place.
OPEN_LOOP = ("torque", "airmass")

MIN_BIN_ROWS = 3
BIN_RPM = 500
MAX_COLUMNS = 3
# Fraction of peak load a row must reach to count as part of the pull.
LOAD_GATE = 0.45


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", SETPOINT_RE.sub(" ", name)).strip().lower()


def find_pairs(cols: list[str]) -> list[tuple[int, int, str]]:
    """Every (actual, setpoint, label) pair the log contains.

    Discovered from the column names, so a logger that writes "Lambda Soll" or
    "MAP Target" works without a code change.
    """
    actuals: dict[str, int] = {}
    for i, name in enumerate(cols):
        if not SETPOINT_RE.search(name):
            actuals.setdefault(_norm(name), i)
    pairs: list[tuple[int, int, str]] = []
    for i, name in enumerate(cols):
        if not SETPOINT_RE.search(name):
            continue
        j = actuals.get(_norm(name))
        if j is not None and j != i:
            pairs.append((j, i, cols[j].strip()))
    return pairs


def _col(cols: list[str], pattern: re.Pattern[str]) -> int | None:
    for i, name in enumerate(cols):
        if pattern.search(name) and not SETPOINT_RE.search(name):
            return i
    return None


def _short(label: str) -> str:
    """"Lambda (λ)" -> "Lambda". Units belong in the header, not every cell."""
    return re.sub(r"\s*[\(\[].*?[\)\]]", "", label).strip()[:11]


def _num(v: float) -> str:
    """Plain decimal, never scientific.

    `%.4g` renders an airmass miss of 1271 as "-1.27e+03", and a model reading
    that alongside "0.0363" on the lambda line treats them as comparable small
    numbers. It called a 53% deviation "small" for exactly this reason. Digits.
    """
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.4f}"


def _pct(err: float, target: float) -> float | None:
    """Error as a percentage of what was commanded, which is the tuner's unit.

    An error of 1271 means nothing on its own - it is huge on airmass and
    invisible on injector flow. Against its own setpoint it is comparable.
    """
    if abs(target) < 1e-6:
        return None
    return err / abs(target) * 100.0


# How far off a loop has to be, as a percentage of its own command, before it is
# worth a word. Below this, control-loop noise.
HOLDS_PCT = 4.0
# Above this it is not a trim, something is genuinely not being delivered.
SERIOUS_PCT = 15.0
# A trend counts as growing/shrinking only if the ends differ by this much.
TREND_RATIO = 1.6


def _verdict(label: str, bands: list[tuple[int, float, float]]) -> tuple[str, str] | None:
    """Judge ONE loop across the whole pull, in code.

    Asked to judge for itself, the model looked at a table showing airmass 53%
    under command and wrote "the errors are all small". That is not a prompting
    problem - magnitude judgement over a column of numbers is arithmetic, and
    everywhere else in this bot arithmetic is done here and handed over as a
    finding. So the verdict is computed and the model reports it.

    `bands` is (rpm, actual, commanded) per bin, in rpm order. Returns
    (kind, sentence) - the kind lets the caller spot loops that all fail the same
    way, which is usually one physical event rather than several faults.
    """
    scored = [
        (rpm, a, t, _pct(a - t, t)) for rpm, a, t in bands if _pct(a - t, t) is not None
    ]
    if len(scored) < 2:
        return None
    worst_rpm, worst_a, worst_t, worst = max(scored, key=lambda x: abs(x[3]))
    if abs(worst) < HOLDS_PCT:
        return "holds", f"{label}: on command throughout (worst {worst:+.0f}%)."

    first, last = scored[0][3], scored[-1][3]
    last_rpm = scored[-1][0]
    where = (
        f"worst {_num(worst_a)} vs {_num(worst_t)} commanded at {worst_rpm} rpm"
        f" ({worst:+.0f}%)"
    )

    # A loop converging from -53% to +5% crosses zero once, and calling that
    # "oscillating" was worse than saying nothing - it is a loop catching up.
    # Oscillation means the error keeps changing its mind, so count the changes,
    # and only after the trend tests have had their say.
    significant = [p for _r, _a, _t, p in scored if abs(p) >= HOLDS_PCT]
    flips = sum(
        1 for x, y in zip(significant, significant[1:]) if (x > 0) != (y > 0)
    )

    # Severity follows the SHAPE, not the peak error. A loop 50% under command at
    # 5000 rpm and back on target by 7500 is a turbo spooling, and labelling that
    # "missing target badly" got a healthy pull called unhealthy.
    big = abs(worst) >= SERIOUS_PCT
    if abs(first) > abs(last) * TREND_RATIO:
        kind = "recovers"
        severity = "low early, recovers"
        shape = f"{first:+.0f}% at {scored[0][0]} closing to {last:+.0f}% by {last_rpm}"
    elif abs(last) > abs(first) * TREND_RATIO:
        kind = "opening"
        severity = "error grows with rpm" + (", badly" if big else "")
        shape = f"{first:+.0f}% at {scored[0][0]} opening to {last:+.0f}% by {last_rpm}"
    elif flips >= 2:
        kind = "hunting"
        severity = f"error changes sign {flips} times"
        shape = "not settling on command"
    else:
        mean_pct = sum(p for _r, _a, _t, p in scored) / len(scored)
        kind = "offset"
        severity = "off the whole way" + (", badly" if big else "")
        shape = f"steady {mean_pct:+.0f}% from {scored[0][0]} to {last_rpm}"
    return kind, f"{label}: {severity}; {where}; {shape}."


def tracking_table(
    rows: list[list[str]],
    cols: list[str],
    to_float,
    roles: dict[str, int] | None = None,
    max_bins: int = 12,
) -> str:
    """Actual vs commanded across the RPM range of the pull.

    `roles` comes from the caller's channel detection. Rolling a fresh one here
    picked "Engagement RPM" - a column of zeros - over "Engine Speed", which is
    exactly the collision the shared detector is ordered to avoid.
    """
    roles = roles or {}
    rpm_i = roles.get("rpm", _col(cols, RPM_RE))
    pairs = find_pairs(cols)
    if rpm_i is None or not pairs:
        return ""
    headline = [p for p in pairs if any(h in p[2].lower() for h in HEADLINE)]
    headline = (headline or pairs)[:MAX_COLUMNS]
    knock_i = roles.get("knock", _col(cols, KNOCK_RE))
    timing_i = roles.get("timing", _col(cols, TIMING_RE))
    load_i = roles.get("boost", roles.get("tps", _col(cols, BOOST_RE)))

    def value(row: list[str], i: int | None) -> float | None:
        return to_float(row[i]) if i is not None and i < len(row) else None

    raw = []
    for row in rows[1:]:
        rpm = value(row, rpm_i)
        if rpm is None or rpm <= 0:
            continue
        raw.append((rpm, row, value(row, load_i)))
    if len(raw) < MIN_BIN_ROWS * 2:
        return ""

    # Restrict to the loaded part of the run. Off load the loops are doing
    # something else entirely - lambda reads 2.0 on fuel cut - and those rows
    # otherwise poison every low-rpm bin.
    loads = [ld for _r, _row, ld in raw if ld is not None]
    gated = raw
    if loads:
        peak = max(loads)
        floor = min(loads) + (peak - min(loads)) * LOAD_GATE
        under_load = [s for s in raw if s[2] is not None and s[2] >= floor]
        if len(under_load) >= MIN_BIN_ROWS * 2:
            gated = under_load

    lo = min(s[0] for s in gated)
    hi = max(s[0] for s in gated)
    if hi - lo < BIN_RPM:
        return ""
    step = max(BIN_RPM, round((hi - lo) / max_bins / BIN_RPM) * BIN_RPM or BIN_RPM)

    buckets: dict[int, list] = {}
    for sample in gated:
        buckets.setdefault(int(sample[0] // step) * step, []).append(sample)

    width = 28
    header = "  rpm    " + "".join(
        f"{_short(lbl) + ' act/cmd err':<{width}}" for _a, _t, lbl in headline
    )
    if timing_i is not None:
        header += "timing  "
    if knock_i is not None:
        header += "worst knock"
    lines = [header, "  " + "-" * (len(header) - 2)]

    for start in sorted(buckets):
        group = buckets[start]
        if len(group) < MIN_BIN_ROWS:
            continue
        cells = ""
        for a_i, t_i, _lbl in headline:
            got = [v for v in (value(r, a_i) for _x, r, _l in group) if v is not None]
            want = [v for v in (value(r, t_i) for _x, r, _l in group) if v is not None]
            if not got or not want:
                cells += f"{'-':<{width}}"
                continue
            a = sum(got) / len(got)
            t = sum(want) / len(want)
            pct = _pct(a - t, t)
            cell = f"{_num(a)}/{_num(t)} {a - t:+.0f}" if abs(a - t) >= 10 else \
                   f"{_num(a)}/{_num(t)} {a - t:+.4f}"
            if pct is not None:
                cell += f" ({pct:+.0f}%)"
            cells += cell.ljust(width)
        line = f"  {start:<7d}{cells}"
        if timing_i is not None:
            tim = [v for v in (value(r, timing_i) for _x, r, _l in group) if v is not None]
            line += (f"{sum(tim) / len(tim):5.1f}   " if tim else "  -     ")
        if knock_i is not None:
            kn = [v for v in (value(r, knock_i) for _x, r, _l in group) if v is not None]
            # Largest magnitude: SimosTools logs retard negative, others positive.
            line += (f"{max(kn, key=abs):+.2f}" if kn else "-")
        lines.append(line.rstrip())
    if len(lines) <= 3:
        return ""

    # Every pair gets judged, not just the three with columns. The columns are
    # only what fits on a line; a loop without one is not a loop nobody cares
    # about, and the summary it used to get - one worst-error number with no
    # denominator - was unreadable as a finding.
    verdicts: list = []
    open_loop: list[str] = []
    for a_i, t_i, label in pairs:
        bands: list[tuple[int, float, float]] = []
        for start in sorted(buckets):
            group = buckets[start]
            if len(group) < MIN_BIN_ROWS:
                continue
            got = [v for v in (value(r, a_i) for _x, r, _l in group) if v is not None]
            want = [v for v in (value(r, t_i) for _x, r, _l in group) if v is not None]
            if got and want:
                bands.append((start, sum(got) / len(got), sum(want) / len(want)))
        got_verdict = _verdict(_short(label), bands)
        if not got_verdict:
            continue
        # Torque request is not a target on a car tuned to PUT SP - the tuner has
        # taken the torque model out of the loop on purpose. Reporting the
        # divergence as a fault led a review with something intentional.
        if any(o in label.lower() for o in OPEN_LOOP):
            if bands:
                worst = max(bands, key=lambda b: abs(b[1] - b[2]))
                open_loop.append(
                    f"{_short(label)} {_num(worst[1])} vs {_num(worst[2])} req at {worst[0]} rpm"
                )
            continue
        verdicts.append((got_verdict[0], got_verdict[1], _short(label), bands))
    rank = {"opening": 0, "offset": 1, "hunting": 2, "recovers": 3, "holds": 4}
    verdicts.sort(key=lambda v: rank.get(v[0], 5))

    tail = ""
    if verdicts or open_loop:
        # Several loops all low early and all recovered is ONE event - the turbo
        # had not spooled - and six near-identical lines bury the ones that matter.
        recovering = [v for v in verdicts if v[0] == "recovers"]
        body: list[str] = ["  " + v[1] for v in verdicts if v[0] not in ("recovers", "holds")]
        if len(recovering) >= 3:
            worst_of = []
            for _k, _line, label, bands in recovering:
                scored = [(rpm, _pct(a - t, t)) for rpm, a, t in bands if _pct(a - t, t)]
                if scored:
                    rpm, pc = max(scored, key=lambda x: abs(x[1]))
                    worst_of.append(f"{label} {pc:+.0f}% at {rpm}")
            body.append(
                f"  {len(recovering)} loops low early and all back on command by the top"
                f" of the pull ({', '.join(worst_of)}) - one event, spool."
            )
        else:
            body += ["  " + v[1] for v in recovering]
        body += ["  " + v[1] for v in verdicts if v[0] == "holds"]
        if open_loop:
            body.append(
                "  open loop by design, not a finding unless they ask: "
                + ", ".join(open_loop)
            )
        tail = "\n\nPER LOOP, computed across the pull:\n" + "\n".join(body)

    return (
        f"TRACKING ACROSS THE PULL - actual/commanded and the error per {step} rpm,\n"
        "loaded rows only. Computed over every row, so these figures are exact.\n"
        + "\n".join(lines)
        + tail
    )
