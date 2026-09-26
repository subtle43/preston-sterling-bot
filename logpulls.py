"""Find the pulls in a datalog and check them against fixed limits.

The review used to treat every row above 45% of peak load as "the pull". A log
with a 2nd and a 3rd gear run, or two separate runs, was averaged into one blur,
and the model then decided for itself what was wrong - which is where "critically
low knock" came from. This does both jobs in code:

  * PULLS: each wide-open-throttle run, split at gear changes, with its gear, rpm
    span, duration and headline figures. The best one is reviewed.
  * FINDINGS: fixed checks per pull (lean under boost, knock per cylinder, boost
    vs target, rail pressure vs setpoint, LPFP, IAT, misfires, fuel trims), each
    tied to the rpm band where it happened. A check that passed says so too, so
    "this log is clean" can be said with specifics instead of invented doubts.

The model narrates FINDINGS; it does not get to add or reverse them.

Column names are matched loosely (SimosTools, VCDS-style and generic loggers);
a check whose channels are missing is simply skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# -- channel detection ---------------------------------------------------------

UNIT_RE = re.compile(r"\(([^)]*)\)\s*$")


def _base(name: str) -> str:
    """Header without its unit, lower case: "Knock Cyl 1 (°)" -> "knock cyl 1"."""
    return " ".join(UNIT_RE.sub("", name).lower().split())


def _unit(name: str) -> str:
    m = UNIT_RE.search(name)
    return m.group(1).strip().lower() if m else ""


_SP = r"(?:sp|set\s*point|setpoint|target|soll|req|cmd|desired|spec)"
PATTERNS: dict[str, re.Pattern[str]] = {
    "time": re.compile(r"^(?:time|timestamp|seconds?|t)$"),
    "rpm": re.compile(r"^(?:engine\s*speed|engine\s*rpm|rpm|nmot|n_eng|drehzahl)$"),
    "pedal": re.compile(r"^(?:pedal(?:\s*pos(?:ition)?)?|accel(?:erator)?\s*pedal.*|app|pvs|pedal\s*%)$"),
    "tps": re.compile(r"^(?:tps|throttle(?:\s*(?:pos(?:ition)?|angle|plate))?|tps\s*%)$"),
    "gear": re.compile(r"^(?:gear|current\s*gear|gear\s*(?:act|actual|engaged))$"),
    "boost": re.compile(r"^boost(?:\s*(?:pressure|actual|act))?$"),
    "put": re.compile(r"^(?:put|pre[\s-]*throttle(?:\s*pressure)?|charge\s*pressure|boost\s*pressure\s*abs)$"),
    "put_sp": re.compile(rf"^(?:put|pre[\s-]*throttle(?:\s*pressure)?|charge\s*pressure)\s*{_SP}$"),
    "map": re.compile(r"^(?:map|manifold\s*(?:abs(?:olute)?\s*)?pressure)$"),
    "map_sp": re.compile(rf"^(?:map|manifold\s*pressure)\s*{_SP}$"),
    "ambient": re.compile(r"^(?:ambient\s*press(?:ure)?|baro(?:metric)?(?:\s*press(?:ure)?)?|atm\w*\s*press\w*)$"),
    "lambda": re.compile(r"^(?:lambda(?:\s*(?:act|actual|bank\s*1))?|afr\s*lambda)$"),
    "lambda_sp": re.compile(rf"^lambda\s*{_SP}$"),
    "knock_avg": re.compile(r"^knock\s*(?:avg|average|retard(?:\s*avg)?)$"),
    "ign": re.compile(r"^(?:ign(?:ition)?\s*(?:avg|average|timing(?:\s*avg)?|angle)|timing(?:\s*avg)?)$"),
    "iat": re.compile(r"^(?:iat|intake\s*air\s*temp\w*|charge\s*air\s*temp\w*)$"),
    "fp_di": re.compile(r"^(?:fp\s*di|rail\s*pressure|hpfp(?:\s*pressure)?|fuel\s*rail\s*pressure|fp\s*hp)$"),
    "fp_di_sp": re.compile(rf"^(?:fp\s*di|rail\s*pressure|hpfp(?:\s*pressure)?|fuel\s*rail\s*pressure|fp\s*hp)\s*{_SP}$"),
    "fp_mpi": re.compile(r"^(?:fp\s*mpi|lpfp\s*pressure|low\s*(?:side\s*)?fuel\s*pressure|fp\s*lp)$"),
    "fp_mpi_sp": re.compile(rf"^(?:fp\s*mpi|lpfp\s*pressure|low\s*(?:side\s*)?fuel\s*pressure|fp\s*lp)\s*{_SP}$"),
    "lpfp_duty": re.compile(r"^lpfp\s*(?:duty|pwm)(?:\s*cycle)?$"),
    "wg": re.compile(r"^(?:wg\s*pos(?:ition)?\s*final|wastegate\s*(?:pos(?:ition)?|duty)|wgdc|wg\s*duty)$"),
    "misfires": re.compile(r"^(?:misfire\s*sum|misfires|misfire\s*count(?:er)?)$"),
    "stft": re.compile(r"^(?:stft|short\s*term\s*fuel\s*trim)$"),
    "ltft": re.compile(r"^(?:ltft|long\s*term\s*fuel\s*trim)$"),
}
KNOCK_CYL_RE = re.compile(r"^(?:knock|knk|ign(?:ition)?\s*retard|knock\s*retard)\s*(?:cyl(?:inder)?\s*)?(\d)$")
MISFIRE_CYL_RE = re.compile(r"^misfire(?:s)?\s*(?:cyl(?:inder)?\s*)?(\d)$")


def detect(cols: list[str]) -> tuple[dict[str, int], dict[int, int]]:
    """Role -> column index, and cylinder -> knock column index.

    Per-cylinder misfire counters go in roles as "misfire_cyl<N>"."""
    roles: dict[str, int] = {}
    knock: dict[int, int] = {}
    for i, raw in enumerate(cols):
        name = _base(raw)
        m = KNOCK_CYL_RE.match(name)
        if m:
            knock.setdefault(int(m.group(1)), i)
            continue
        m = MISFIRE_CYL_RE.match(name)
        if m:
            roles.setdefault(f"misfire_cyl{m.group(1)}", i)
            continue
        for role, pat in PATTERNS.items():
            if role not in roles and pat.match(name):
                roles[role] = i
                break
    return roles, knock


# -- pulls -----------------------------------------------------------------------

WOT_PEDAL = 80.0          # % pedal counts as wide open
WOT_TPS = 70.0
MIN_PULL_S = 1.0
MIN_PULL_RPM = 800
MIN_PEAK_RPM = 3000
GAP_S = 0.3               # a flicker off the pedal shorter than this does not end a pull
SHIFT_DROP_RPM = 600      # rpm falling this much means a shift, i.e. a new pull
SPOOL_RPM = 2800          # below this, missing boost target is spool-up, not a fault


@dataclass
class Pull:
    n: int
    rows: list[int]                       # indexes into the data rows
    gear: int | None = None
    t0: float = 0.0
    t1: float = 0.0
    rpm0: float = 0.0
    rpm1: float = 0.0
    summary: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.t1 - self.t0

    @property
    def span(self) -> float:
        return self.rpm1 - self.rpm0


def _series(rows: list[list[str]], i: int | None, to_float) -> list[float | None]:
    if i is None:
        return [None] * len(rows)
    return [to_float(r[i]) if i < len(r) else None for r in rows]


def find_pulls(data: list[list[str]], roles: dict[str, int], to_float) -> tuple[list[Pull], list[float | None]]:
    rpm = _series(data, roles.get("rpm"), to_float)
    t = _series(data, roles.get("time"), to_float)
    if all(v is None for v in t):
        t = [i * 0.05 for i in range(len(data))]       # assume 20 Hz without a clock
    else:
        # Some loggers write milliseconds. A median step over 5 cannot be seconds.
        steps = sorted(b - a for a, b in zip(t, t[1:]) if a is not None and b is not None and b > a)
        if steps and steps[len(steps) // 2] > 5:
            t = [v / 1000 if v is not None else None for v in t]
    if "pedal" in roles:
        gate = [v is not None and v >= WOT_PEDAL for v in _series(data, roles["pedal"], to_float)]
    elif "tps" in roles:
        gate = [v is not None and v >= WOT_TPS for v in _series(data, roles["tps"], to_float)]
    else:
        # No pedal or throttle: fall back to the load channel near its peak.
        load_i = next((roles[k] for k in ("put", "boost", "map") if k in roles), None)
        load = _series(data, load_i, to_float)
        vals = [v for v in load if v is not None]
        if not vals:
            return [], t
        lo, hi = min(vals), max(vals)
        gate = [v is not None and v >= lo + (hi - lo) * 0.7 for v in load]
    gear = _series(data, roles.get("gear"), to_float)

    segments: list[list[int]] = []
    cur: list[int] = []
    last_on: float | None = None
    for i in range(len(data)):
        if rpm[i] is None or t[i] is None:
            continue
        if gate[i]:
            shift = bool(cur) and (
                (gear[i] is not None and gear[cur[-1]] is not None and gear[i] != gear[cur[-1]])
                or (rpm[cur[-1]] is not None and rpm[cur[-1]] - rpm[i] >= SHIFT_DROP_RPM)
            )
            gap = last_on is not None and t[i] - last_on > GAP_S
            if cur and (shift or gap):
                segments.append(cur)
                cur = []
            cur.append(i)
            last_on = t[i]
    if cur:
        segments.append(cur)

    pulls: list[Pull] = []
    for seg in segments:
        rp = [rpm[i] for i in seg if rpm[i] is not None]
        if not rp:
            continue
        p = Pull(n=0, rows=seg, t0=t[seg[0]], t1=t[seg[-1]], rpm0=min(rp[: max(1, len(rp) // 5)]), rpm1=max(rp))
        if p.duration < MIN_PULL_S or p.span < MIN_PULL_RPM or p.rpm1 < MIN_PEAK_RPM:
            continue
        gears = [int(g) for g in (gear[i] for i in seg) if g is not None and g > 0]
        p.gear = max(set(gears), key=gears.count) if gears else None
        pulls.append(p)
    for n, p in enumerate(pulls, 1):
        p.n = n
    return pulls, t


def best_pull(pulls: list[Pull]) -> Pull | None:
    """The run most worth reviewing: widest rpm span, 3rd gear and up preferred."""
    if not pulls:
        return None
    return max(pulls, key=lambda p: p.span * (1.25 if (p.gear or 0) >= 3 else 1.0) + p.duration * 50)


# -- checks -----------------------------------------------------------------------

# Simos 18 commands lambda 1.0 well into boost on purpose, so stoich as commanded
# is not lean. Leaner than COMMANDED is the fault; the absolute limit is only a
# fallback for logs with no lambda setpoint.
LEAN_LAMBDA = 0.95
LEAN_OVER_SP = 0.05
SUSTAIN_S = 0.3           # a check has to hold this long - one-row transients are not findings
KNOCK_DEG = 1.5
KNOCK_BAD_DEG = 3.0
BOOST_SHORT_PCT = 8.0
BOOST_OVER_PCT = 10.0
RAIL_SHORT_PCT = 10.0
LPFP_SHORT_PCT = 15.0
LPFP_DUTY_MAX = 95.0
WG_MAX = 95.0
TRIM_PCT = 10.0
MIN_SAMPLES = 3


def _temp_limits(unit: str) -> tuple[float, float]:
    """(rise worth mentioning, absolute hot) in the log's own unit."""
    return (20.0, 140.0) if "f" in unit else (11.0, 60.0)


def _pressure_to_psi(unit: str) -> float:
    return {"kpa": 0.145038, "bar": 14.5038, "mbar": 0.0145038, "hpa": 0.0145038, "psi": 1.0}.get(unit, 0.0)


def _bands(rpms: list[float]) -> str:
    """"4800-5600 rpm" for the samples that tripped a check."""
    if not rpms:
        return ""
    lo, hi = int(min(rpms) // 100 * 100), int(-(-max(rpms) // 100) * 100)
    return f"{lo} rpm" if hi - lo <= 200 else f"{lo}-{hi} rpm"


def _runs(flags: list[bool]) -> int:
    """Longest run of consecutive True - sustained, not a single glitch."""
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def _sustained(flags: list[bool], t: list[float | None]) -> bool:
    """True held for SUSTAIN_S seconds (and at least MIN_SAMPLES rows).

    Sample-count alone meant 0.1 s at 30 Hz - a launch's lambda wobble passed
    as "lean under boost".
    """
    start = None
    run = 0
    for f, ts in zip(flags, t):
        if f and ts is not None:
            if start is None:
                start, run = ts, 0
            run += 1
            if run >= MIN_SAMPLES and ts - start >= SUSTAIN_S:
                return True
        elif not f:
            start, run = None, 0
    return False


def check_pull(p: Pull, data, cols, roles, knock_cols, to_float, t_all, misfire_cols=None) -> tuple[list[tuple[str, str]], dict]:
    """(severity, text) findings for one pull, plus its headline summary."""
    rows = [data[i] for i in p.rows]
    t = [t_all[i] for i in p.rows]

    def col(role: str) -> list[float | None]:
        return _series(rows, roles.get(role), to_float)

    def held(flags: list[bool]) -> bool:
        return _sustained(flags, t)

    rpm = col("rpm")
    found: list[tuple[str, str]] = []
    summary: dict = {}

    # Boost, as psi above ambient where possible.
    boost_psi: list[float | None] = [None] * len(rows)
    if "boost" in roles and _unit(cols[roles["boost"]]) in ("psi", "psig", ""):
        boost_psi = col("boost")
    elif "put" in roles:
        f = _pressure_to_psi(_unit(cols[roles["put"]]))
        amb = col("ambient") if "ambient" in roles else [None] * len(rows)
        if f:
            boost_psi = [
                (v - (a if a is not None else 101.3 / (f / 0.145038))) * f if v is not None else None
                for v, a in zip(col("put"), amb)
            ]
    pb = [(b, i) for i, b in enumerate(boost_psi) if b is not None]
    under_boost = [b is not None and b >= 8 for b in boost_psi]
    if pb:
        peak, at = max(pb)
        summary["peak_boost"] = f"{peak:.1f} psi"
        lam = col("lambda")
        if lam[at] is not None:
            summary["lambda_at_peak"] = f"{lam[at]:.2f}"

    # Lean under boost. With a setpoint, the fault is running leaner than
    # commanded; stoich as commanded is how Simos 18 is calibrated.
    lam = col("lambda")
    sp = col("lambda_sp")
    has_sp = any(v is not None for v in sp)
    if any(v is not None for v in lam):
        boosted = [(v, s) for u, v, s in zip(under_boost, lam, sp) if u and v is not None and v < 1.3]
        if has_sp:
            off = [u and a is not None and s is not None and a < 1.3 and a - s > LEAN_OVER_SP
                   for u, a, s in zip(under_boost, lam, sp)]
            if held(off):
                gap = max(a - s for f, a, s in zip(off, lam, sp) if f)
                found.append(("HIGH", f"LEANER THAN COMMANDED UNDER BOOST: lambda ran up to {gap:.2f} leaner "
                                      f"than its setpoint for over {SUSTAIN_S:g} s at "
                                      f"{_bands([r for f, r in zip(off, rpm) if f and r])} - the fuel system or "
                                      "the fuelling model is not delivering what the tune asks for."))
            elif boosted:
                found.append(("OK", f"Fuelling under boost: lambda {min(v for v, _ in boosted):.2f}-"
                                    f"{max(v for v, _ in boosted):.2f}, following its setpoint "
                                    f"(commanded {min(s for _, s in boosted if s is not None):.2f}-"
                                    f"{max(s for _, s in boosted if s is not None):.2f})."))
        else:
            lean = [u and v is not None and LEAN_LAMBDA < v < 1.3 for u, v in zip(under_boost, lam)]
            if held(lean):
                worst = max(v for f, v in zip(lean, lam) if f)
                found.append(("HIGH", f"LEAN UNDER BOOST: lambda reached {worst:.2f} with 8+ psi of boost at "
                                      f"{_bands([r for f, r in zip(lean, rpm) if f and r])} (no lambda setpoint "
                                      "logged to compare against)."))
            elif boosted:
                found.append(("OK", f"Fuelling under boost: lambda {min(v for v, _ in boosted):.2f}-"
                                    f"{max(v for v, _ in boosted):.2f}."))

    # Knock, per cylinder.
    if knock_cols:
        per: dict[int, list[float]] = {}
        worst_cyl, worst_val = None, 0.0
        for cyl, ci in sorted(knock_cols.items()):
            vals = [abs(v) if v is not None else 0.0 for v in _series(rows, ci, to_float)]
            per[cyl] = vals
            if vals and max(vals) > worst_val:
                worst_cyl, worst_val = cyl, max(vals)
        summary["worst_knock"] = f"{worst_val:.1f}° (cyl {worst_cyl})" if worst_cyl else "0°"
        hot = {c: v for c, v in per.items() if held([x >= KNOCK_DEG for x in v]) or max(v) >= KNOCK_BAD_DEG}
        if hot:
            where = _bands([r for c, v in hot.items() for x, r in zip(v, rpm) if x >= KNOCK_DEG and r])
            cyls = ", ".join(f"cyl {c} {max(v):.1f}°" for c, v in hot.items())
            if len(hot) == 1 and len(per) >= 3:
                why = ("only ONE cylinder - that points at something local to it (plug gap, coil, "
                       "injector, carbon) rather than the tune as a whole")
            elif len(hot) == len(per):
                why = "ALL cylinders - timing or boost is past what this fuel and these temperatures allow"
            else:
                why = "more than one cylinder"
            sev = "HIGH" if max(max(v) for v in hot.values()) >= KNOCK_BAD_DEG else "MED"
            found.append((sev, f"KNOCK: {cyls} pulled at {where} - {why}."))
        else:
            found.append(("OK", f"Knock: every cylinder stayed under {KNOCK_DEG}° sustained "
                                f"(worst single sample {worst_val:.1f}°)."))
    elif "knock_avg" in roles:
        vals = [abs(v) for v in col("knock_avg") if v is not None]
        if vals:
            summary["worst_knock"] = f"{max(vals):.1f}° (average)"

    # Boost against its target, above spool. PUT/PUT SP is the pair a tuned car
    # actually controls boost with; MAP/MAP SP is the fallback.
    for act, tgt, label in (("put", "put_sp", "PUT"), ("map", "map_sp", "MAP")):
        if act not in roles or tgt not in roles:
            continue
        a, s, wg = col(act), col(tgt), col("wg")
        # Judge tracking only once boost has first come up to its target: before
        # that the turbo is spooling, and a 1st-gear launch reads "66% short".
        spooled_at = next((k for k, (r, x, y) in enumerate(zip(rpm, a, s))
                           if r and r >= SPOOL_RPM and x is not None and y and x >= y * 0.92), None)
        if spooled_at is None:
            reached = [(x, y) for x, y in zip(a, s) if x is not None and y]
            if reached:
                x_pk = max(x for x, _ in reached)
                y_pk = max(y for _, y in reached)
                found.append(("MED", f"BOOST NEVER REACHED TARGET: {label} peaked at {x_pk:.0f} against a "
                                     f"setpoint of {y_pk:.0f} ({(x_pk - y_pk) / y_pk * 100:.0f}%) - either the "
                                     "pull was too short to spool, or the target is beyond this turbo."))
            break
        short, over, wg_at_short = [], [], []
        for k, (r, x, y, w) in enumerate(zip(rpm, a, s, wg)):
            if k < spooled_at or r is None or x is None or not y:
                short.append(False); over.append(False); continue
            err = (x - y) / y * 100
            short.append(err <= -BOOST_SHORT_PCT)
            over.append(err >= BOOST_OVER_PCT)
            if err <= -BOOST_SHORT_PCT and w is not None:
                wg_at_short.append(w)
        errs = [(x - y) / y * 100 for k, (x, y) in enumerate(zip(a, s)) if k >= spooled_at and x is not None and y]
        if held(short):
            worst = min(errs)
            where = _bands([r for f, r in zip(short, rpm) if f and r])
            if wg_at_short and max(wg_at_short) >= WG_MAX:
                cause = (f"with the wastegate at {max(wg_at_short):.0f}% - it is out of authority, so the "
                         "turbo cannot make more; ask for less up there or it is a hardware limit")
            elif wg_at_short:
                cause = (f"with the wastegate only at {max(wg_at_short):.0f}% - it had room left, so this "
                         "is boost control (feed-forward or PID), not the turbo running out")
            else:
                cause = "(no wastegate channel logged to say why)"
            found.append(("MED", f"BOOST SHORT OF TARGET: {label} fell up to {abs(worst):.0f}% under its "
                                 f"setpoint at {where}, {cause}."))
        elif held(over):
            found.append(("MED", f"BOOST OVERSHOOT: {label} ran up to {max(errs):.0f}% over its setpoint at "
                                 f"{_bands([r for f, r in zip(over, rpm) if f and r])}."))
        elif errs:
            found.append(("OK", f"Boost tracking: once spooled, {label} held within "
                                f"{max(abs(e) for e in errs):.0f}% of target."))
        break

    # High-pressure rail against its setpoint.
    if "fp_di" in roles and "fp_di_sp" in roles:
        a, s = col("fp_di"), col("fp_di_sp")
        short = [x is not None and y and (y - x) / y * 100 >= RAIL_SHORT_PCT for x, y in zip(a, s)]
        errs = [(y - x) / y * 100 for x, y in zip(a, s) if x is not None and y]
        if held(short):
            found.append(("HIGH", f"HPFP FALLING BEHIND: direct-injection rail pressure dropped up to "
                                  f"{max(errs):.0f}% under its setpoint at "
                                  f"{_bands([r for f, r in zip(short, rpm) if f and r])} - the pump cannot "
                                  "supply this fuel flow, and lean follows."))
        elif errs:
            found.append(("OK", f"HPFP: rail pressure held within {max(0.0, max(errs)):.0f}% of setpoint."))

    # Low-pressure side.
    if "fp_mpi" in roles and "fp_mpi_sp" in roles:
        a, s = col("fp_mpi"), col("fp_mpi_sp")
        short = [x is not None and y and (y - x) / y * 100 >= LPFP_SHORT_PCT for x, y in zip(a, s)]
        if held(short):
            worst = max((y - x) / y * 100 for f, x, y in zip(short, a, s) if f)
            found.append(("HIGH", f"LOW-PRESSURE FUEL SAGGING: low-side pressure fell up to {worst:.0f}% "
                                  f"under target at {_bands([r for f, r in zip(short, rpm) if f and r])} - "
                                  "the LPFP is not keeping up, and it starves the HPFP."))
    if "lpfp_duty" in roles:
        d = [v for v in col("lpfp_duty") if v is not None]
        if d and max(d) >= LPFP_DUTY_MAX:
            found.append(("MED", f"LPFP MAXED: pump duty hit {max(d):.0f}% - no headroom left for more fuel."))

    # Intake air temperature.
    if "iat" in roles:
        iat = [v for v in col("iat") if v is not None]
        if iat:
            unit = _unit(cols[roles["iat"]])
            rise, hot = _temp_limits(unit)
            deg = "°F" if "f" in unit else "°C"
            summary["iat"] = f"{iat[0]:.0f}→{iat[-1]:.0f}{deg}"
            if iat[-1] - iat[0] >= rise:
                found.append(("MED", f"HEAT SOAK: IAT climbed {iat[-1] - iat[0]:.0f}{deg} across the pull "
                                     f"({iat[0]:.0f}→{iat[-1]:.0f}{deg}) - the intercooler is not keeping up."))
            elif max(iat) >= hot:
                found.append(("MED", f"HOT INTAKE: IAT reached {max(iat):.0f}{deg}."))

    # Misfires counted during the pull. Per-cylinder counters when logged: the
    # SimosTools "Misfires" channel jumped to 77 on a pull where the cylinder
    # counters showed exactly one misfire, on cylinder 1.
    if misfire_cols:
        hits = {}
        for cyl, ci in sorted(misfire_cols.items()):
            m = [v for v in _series(rows, ci, to_float) if v is not None]
            if m and max(m) - min(m) >= 1:
                hits[cyl] = max(m) - min(m)
        if hits:
            # A count or two in a hard launch can be rough-road false detection;
            # a handful on one cylinder is a real ignition or fuel fault.
            total = sum(hits.values())
            sev = "HIGH" if total >= 5 else "MED"
            tail = (" - one cylinder points at its plug, coil or injector" if len(hits) == 1
                    else " - spread across cylinders")
            if total < 5:
                tail += "; only a few counts, so see whether it repeats on the same cylinder next log"
            found.append((sev, "MISFIRES during the pull: " + ", ".join(
                f"cyl {c} +{n:.0f}" for c, n in hits.items()) + tail + "."))
        else:
            found.append(("OK", "Misfires: none counted on any cylinder during the pull."))
    elif "misfires" in roles:
        m = [v for v in col("misfires") if v is not None]
        if m and max(m) - min(m) >= 1:
            found.append(("MED", f"MISFIRES: the misfire counter rose by {max(m) - min(m):.0f} during the "
                                 "pull (no per-cylinder counters logged)."))

    # Fuel trims pinned far from zero.
    for role, label in (("stft", "short-term"), ("ltft", "long-term")):
        if role in roles:
            series = col(role)
            v = [x for x in series if x is not None]
            # Short-term trim flicks past 10% on every transient; only a trim that
            # stays there says the fuelling model is off.
            if v and held([x is not None and abs(x) >= TRIM_PCT for x in series]):
                worst = max(v, key=abs)
                found.append(("MED", f"FUEL TRIM: {label} trim reached {worst:+.0f}% - the fuelling model is "
                                     "off by that much and the ECU is correcting for it."))

    if "ign" in roles:
        ign = [v for v in col("ign") if v is not None]
        if ign:
            summary["peak_timing"] = f"{max(ign):.1f}°"
    return found, summary


# A logger's channel list (SimosTools / ecu-hsl PID lists) is posted as often as
# a log, and it is a CSV too. Reviewed as a log it produced a confident review of
# nothing.
PID_LIST_RE = re.compile(r"^name$|^unit$|^equation$|^address$|^format$", re.I)
PID_LIST_NOTE = (
    "THIS IS NOT A DATALOG. It is a logger PID LIST - the channel configuration a "
    "logger reads (name, unit, equation, address per channel). There is no recorded "
    "data in it, so there is nothing to review: no boost, no knock, no pull. Say so, "
    "comment on the channel choice if asked, and ask for an actual log recorded with it."
)


def is_pid_list(cols: list[str]) -> bool:
    return sum(bool(PID_LIST_RE.match(c.strip())) for c in cols[:8]) >= 3


def analyze(rows: list[list[str]], cols: list[str], to_float) -> tuple[str, list[list[str]] | None]:
    """(PULLS + FINDINGS block, the best pull's rows with the header first).

    The rows are for the tracking table, so it bins one real pull instead of
    everything above a load threshold.
    """
    if is_pid_list(cols):
        return PID_LIST_NOTE, None
    roles, knock_cols = detect(cols)
    misfire_cols = {int(k[len("misfire_cyl"):]): v for k, v in roles.items() if k.startswith("misfire_cyl")}
    if "rpm" not in roles:
        return "", None
    data = rows[1:]
    pulls, t_all = find_pulls(data, roles, to_float)
    if not pulls:
        # No pull is not no review: a "knock at 2k cruise" log has no pull by
        # design, and knock and misfires still mean something at part throttle.
        whole = Pull(n=0, rows=[i for i in range(len(data)) if t_all[i] is not None])
        found, _summary = check_pull(whole, data, cols, roles, knock_cols, to_float, t_all, misfire_cols)
        keep = [(s, x) for s, x in found if x.startswith(("KNOCK", "Knock", "MISFIRE", "Misfire", "FUEL TRIM"))]
        lines = ["PULLS FOUND: none. There is no wide-open-throttle run of at least "
                 f"{MIN_PULL_S:g} s and {MIN_PULL_RPM} rpm reaching {MIN_PEAK_RPM} rpm - this is idle, "
                 "cruise or part throttle. Boost and power cannot be judged from it; if that is what "
                 "they asked about, ask for a 3rd gear pull from about 2500 rpm to redline, pedal "
                 "to the floor. What the whole log does show:"]
        if keep:
            lines += [f"  [{'OK  ' if s == 'OK' else f'{s:<4}'}] whole log: {x.replace(' during the pull', ' in this log')}"
                      for s, x in keep]
        else:
            lines.append("  (no knock or misfire channels logged)")
        return "\n".join(lines), None
    best = best_pull(pulls)
    lines = ["PULLS FOUND (wide-open-throttle runs, found by code):"]
    all_findings: list[tuple[int, str, str]] = []
    for p in pulls:
        found, summary = check_pull(p, data, cols, roles, knock_cols, to_float, t_all, misfire_cols)
        p.summary = summary
        gear = f"gear {p.gear}, " if p.gear else ""
        bits = [f"{gear}{p.rpm0:.0f}→{p.rpm1:.0f} rpm over {p.duration:.1f} s"]
        for key, label in (("peak_boost", "peak boost"), ("lambda_at_peak", "lambda at peak boost"),
                           ("worst_knock", "worst knock"), ("peak_timing", "peak timing"), ("iat", "IAT")):
            if key in summary:
                bits.append(f"{label} {summary[key]}")
        mark = "   <- REVIEW THIS ONE" if p is best else ""
        lines.append(f"  Pull {p.n}: " + ", ".join(bits) + mark)
        for sev, text in found:
            all_findings.append((p.n, sev, text))

    order = {"HIGH": 0, "MED": 1, "OK": 2}
    # Problems from every pull; the "passed" lines only for the reviewed one.
    shown = [f for f in all_findings if f[1] != "OK" or f[0] == best.n]
    shown.sort(key=lambda f: (order[f[1]], f[0] != best.n, f[0]))
    lines.append("")
    lines.append("FINDINGS - checked by code against fixed limits. These are facts: build the "
                 "review on them, most severe first. Do not add a problem that is not listed here, "
                 "and do not dispute an OK line.")
    if not any(sev != "OK" for _n, sev, _t in shown):
        lines.append(f"  No problems found in any pull. Checks that passed on pull {best.n}:")
    for n, sev, text in shown:
        tag = "OK  " if sev == "OK" else f"{sev:<4}"
        lines.append(f"  [{tag}] pull {n}: {text}")
    header = rows[0]
    best_rows = [header] + [data[i] for i in best.rows]
    return "\n".join(lines), best_rows
