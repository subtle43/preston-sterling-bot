"""A fake dyno sheet for /dyno: a believable chassis-dyno printout of a
member's car, built from a spec the model writes out of their chat history.

The comedy is in the realism: a proper power/torque plot, SAE-style header
block and run notes, with the dips labelled by what actually went wrong
("boost leak (again)") and a dashed curve for what they claimed on Discord.

Object API + Agg canvas only (no pyplot): this runs in a worker thread.
"""

from __future__ import annotations

import io
import math
import random

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

PAPER = "#f7f5ef"
INK = "#1d1f22"
MUTED = "#6b6f76"
GRID = "#d9d6cc"
HP = "#c8102e"
TQ = "#1f4e9c"
CLAIM = "#8a8f98"
NOTE_BG = "#fffbe6"


def _clamp(v, lo, hi, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _shape(rpm: float, start: float, tq_peak: float, end: float) -> float:
    """Turbo-car torque shape, 0..1: spools in, plateaus, then falls away."""
    if rpm <= tq_peak:
        x = (rpm - start) / max(1.0, tq_peak - start)
        return 0.45 + 0.55 * (1 - (1 - max(0.0, x)) ** 2.2)
    x = (rpm - tq_peak) / max(1.0, end - tq_peak)
    return 1.0 - 0.38 * x ** 1.6


def curves(spec: dict) -> dict:
    start = _clamp(spec.get("rpm_start"), 1500, 4000, 2500)
    end = _clamp(spec.get("rpm_end"), start + 1500, 9000, 6800)
    tq_peak = _clamp(spec.get("tq_peak_rpm"), start + 300, end - 500, start + (end - start) * 0.3)
    peak_whp = _clamp(spec.get("peak_whp"), 20, 1500, 250)
    claimed = _clamp(spec.get("claimed_whp"), 0, 2500, 0)
    rpm = [start + (end - start) * i / 399 for i in range(400)]
    rng = random.Random(str(spec.get("car", "")))
    noise = [1 + rng.uniform(-0.006, 0.006) for _ in rpm]
    base_tq = [_shape(r, start, tq_peak, end) for r in rpm]
    # Scale so peak horsepower lands where the spec says.
    raw_hp = [t * r / 5252 for t, r in zip(base_tq, rpm)]
    k = peak_whp / max(raw_hp)
    clean_tq = [t * k for t in base_tq]

    events = []
    for ev in (spec.get("events") or [])[:4]:
        at = _clamp(ev.get("rpm"), start, end, None)
        if at is None:
            continue
        drop = _clamp(ev.get("drop_pct"), 3, 80, 15) / 100
        cliff = bool(ev.get("cliff"))
        events.append((at, drop, cliff, str(ev.get("label", ""))[:48]))

    tq = []
    for r, t, n in zip(rpm, clean_tq, noise):
        f = 1.0
        for at, drop, cliff, _l in events:
            if cliff:
                if r >= at:
                    f *= 1 - drop * min(1.0, (r - at) / 250)
            else:
                f *= 1 - drop * math.exp(-((r - at) / 170) ** 2)
        tq.append(t * f * n)
    hp = [t * r / 5252 for t, r in zip(tq, rpm)]
    claim_hp = [h * claimed / peak_whp for h in (t * r / 5252 for t, r in zip(clean_tq, rpm))] if claimed > peak_whp else None
    return {"rpm": rpm, "hp": hp, "tq": tq, "claim_hp": claim_hp, "events": events}


def render(spec: dict, owner: str) -> bytes:
    c = curves(spec)
    rpm, hp, tq = c["rpm"], c["hp"], c["tq"]
    fig = Figure(figsize=(11, 7.6), dpi=100, facecolor=PAPER)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0.07, 0.2, 0.86, 0.6))
    ax.set_facecolor(PAPER)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.plot(rpm, hp, color=HP, linewidth=2.4, label="Power (whp)")
    ax.plot(rpm, tq, color=TQ, linewidth=2.4, label="Torque (lb-ft)")
    if c["claim_hp"]:
        ax.plot(rpm, c["claim_hp"], color=CLAIM, linewidth=1.6, linestyle="--",
                label="Power as claimed in Discord")
    top = max(max(hp), max(tq), max(c["claim_hp"] or [0]))
    ax.set_ylim(0, top * 1.18)
    ax.set_xlim(rpm[0], rpm[-1])
    ax.set_xlabel("Engine speed (rpm)", color=INK, fontsize=10)
    ax.set_ylabel("whp  /  lb-ft", color=INK, fontsize=10)

    # Label each disaster where it happened, on the power curve.
    for i, (at, _drop, cliff, label) in enumerate(c["events"]):
        if not label:
            continue
        j = min(range(len(rpm)), key=lambda k: abs(rpm[k] - (at + (180 if cliff else 0))))
        y = hp[j]
        ax.annotate(label, xy=(rpm[j], y), xytext=(rpm[j], y + top * (0.16 + 0.07 * (i % 2))),
                    ha="center", fontsize=9, color=INK,
                    bbox={"boxstyle": "round,pad=0.25", "fc": NOTE_BG, "ec": MUTED, "lw": 0.8},
                    arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1})

    pk_hp = max(hp)
    pk_hp_rpm = rpm[hp.index(pk_hp)]
    pk_tq = max(tq)
    pk_tq_rpm = rpm[tq.index(pk_tq)]
    leg = ax.legend(loc="upper left", fontsize=9, facecolor=PAPER, edgecolor=MUTED)
    leg.get_frame().set_alpha(0.95)

    shop = str(spec.get("shop") or "PRESTON STERLING DYNO & DISAPPOINTMENT")[:60]
    fig.text(0.07, 0.95, shop.upper(), fontsize=15, fontweight="bold", color=INK, family="monospace")
    fig.text(0.07, 0.915, f"Customer: {owner}   ·   Vehicle: {str(spec.get('car', 'unknown shitbox'))[:70]}",
             fontsize=10, color=INK)
    fig.text(0.07, 0.885, f"{str(spec.get('run_name', 'Run #1'))[:80]}   ·   "
                          f"Correction: {str(spec.get('correction', 'SAE J1349'))[:30]}   ·   "
                          f"Smoothing: 5   ·   {str(spec.get('dyno', 'Dynojet 224xLC'))[:30]}",
             fontsize=9, color=MUTED)
    fig.text(0.93, 0.95, f"{pk_hp:.1f} whp", fontsize=20, fontweight="bold", color=HP, ha="right")
    fig.text(0.93, 0.905, f"@ {pk_hp_rpm:.0f} rpm", fontsize=9, color=MUTED, ha="right")
    fig.text(0.93, 0.87, f"{pk_tq:.1f} lb-ft @ {pk_tq_rpm:.0f} rpm", fontsize=10, color=TQ, ha="right")

    note = str(spec.get("operator_note") or "").strip()[:230]
    if note:
        fig.text(0.07, 0.085, "Operator notes:", fontsize=9, color=MUTED, fontweight="bold")
        fig.text(0.07, 0.035, _wrap(note, 125), fontsize=10, color=INK, va="bottom")
    fig.text(0.93, 0.012, "Results not typical. Results not possible. Not a real dyno.",
             fontsize=7.5, color=MUTED, ha="right")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=PAPER)
    return buf.getvalue()


def _wrap(text: str, width: int) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines[:2])
