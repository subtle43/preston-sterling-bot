"""A chart of the reviewed pull, posted under the log review.

Stacked panels on one x axis - boost vs target (wastegate on a second axis),
lambda vs setpoint, timing with knock retard per cylinder, HPFP rail vs setpoint -
each drawn only when its channels were logged. Bands where FINDINGS flagged a
problem are shaded, so the picture points at what the review says.

Uses matplotlib's object API with the Agg canvas, never pyplot: pyplot keeps
global state and this runs in a worker thread.
"""

from __future__ import annotations

import io
import re

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import logpulls

BG = "#2b2d31"          # Discord dark
PANEL = "#313338"
INK = "#dbdee1"
MUTED = "#949ba4"
GRID = "#3f4147"
ACTUAL = "#5da9e9"
TARGET = "#f0b429"
EXTRA = "#a78bfa"
KNOCK = ["#f23f43", "#ff8a65", "#ffd166", "#e879f9", "#f97316", "#fb7185"]
HIGH_BAND = "#f23f43"
MED_BAND = "#f0b429"

BAND_RE = re.compile(r"\[(HIGH|MED)\s*\] pull (\d+):.*?at (\d+)(?:-(\d+))? rpm")
REVIEW_RE = re.compile(r"Pull (\d+): (.*?)\s+<- REVIEW THIS ONE")


def _col(rows, i, to_float):
    return [to_float(r[i]) if i is not None and i < len(r) else None for r in rows]


def _gauge_psi(values, unit, ambient):
    """Absolute pressure -> psi above ambient."""
    f = {"kpa": 0.145038, "bar": 14.5038, "mbar": 0.0145038, "hpa": 0.0145038, "psi": 1.0}.get(unit)
    if not f:
        return None
    out = []
    for v, a in zip(values, ambient):
        if v is None:
            out.append(None)
            continue
        base = a if a is not None else 101.3 / (f / 0.145038)
        out.append((v - base) * f)
    return out


def _clean(xs, ys, lo=None, hi=None):
    """Drop Nones (and values outside lo..hi) pairwise, for plotting."""
    px, py = [], []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        if (lo is not None and y < lo) or (hi is not None and y > hi):
            y = float("nan")
        px.append(x)
        py.append(y)
    return px, py


def _style(ax, ylabel):
    ax.set_facecolor(PANEL)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)


def render(pull_rows: list[list[str]], cols: list[str], to_float, filename: str, block: str) -> bytes | None:
    """PNG bytes for the reviewed pull, or None when there is nothing to draw."""
    roles, knock_cols = logpulls.detect(cols)
    rows = pull_rows[1:]
    if len(rows) < 5 or "rpm" not in roles:
        return None
    rpm = _col(rows, roles["rpm"], to_float)
    t = _col(rows, roles.get("time"), to_float)

    # rpm on x reads like a dyno sheet, but a launch with wheelspin runs rpm up
    # and down and the lines fold back on themselves - then time is the axis.
    steps = [b - a for a, b in zip(rpm, rpm[1:]) if a is not None and b is not None]
    rising = sum(s >= 0 for s in steps) / max(1, len(steps))
    if rising >= 0.85:
        x, xlabel, by_rpm = rpm, "engine speed (rpm)", True
        # Keep only rows where rpm is still climbing, like a dyno sheet: at the
        # limiter rpm flutters and every line scribbled back over itself.
        peak = -1.0
        keep = []
        for k, r in enumerate(rpm):
            if r is not None and r >= peak - 40:
                keep.append(k)
                peak = max(peak, r)
        rows = [rows[k] for k in keep]
        rpm = [rpm[k] for k in keep]
        t = [t[k] for k in keep]
        x = rpm
    else:
        t0 = next((v for v in t if v is not None), 0.0)
        x = [v - t0 if v is not None else None for v in t]
        steps_t = sorted(b - a for a, b in zip(x, x[1:]) if a is not None and b is not None and b > a)
        if steps_t and steps_t[len(steps_t) // 2] > 5:          # milliseconds
            x = [v / 1000 if v is not None else None for v in x]
        xlabel, by_rpm = "time into pull (s)", False

    unit = lambda role: logpulls._unit(cols[roles[role]]) if role in roles else ""
    ambient = _col(rows, roles.get("ambient"), to_float)
    panels = []

    # 1. Boost vs target.
    boost = tgt = None
    if "put" in roles and "put_sp" in roles:
        boost = _gauge_psi(_col(rows, roles["put"], to_float), unit("put"), ambient)
        tgt = _gauge_psi(_col(rows, roles["put_sp"], to_float), unit("put_sp"), ambient)
        blabel = "PUT"
    if boost is None and "boost" in roles:
        boost, blabel = _col(rows, roles["boost"], to_float), "boost"
    if boost is not None:
        panels.append("boost")
    if "lambda" in roles:
        panels.append("lambda")
    if "ign" in roles or knock_cols:
        panels.append("timing")
    if "fp_di" in roles and "fp_di_sp" in roles:
        panels.append("rail")
    if "fp_mpi" in roles and "fp_mpi_sp" in roles:
        panels.append("lowside")
    if not panels:
        return None

    fig = Figure(figsize=(11, 2.35 * len(panels) + 0.8), dpi=100, facecolor=BG)
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(panels), 1, sharex=True, squeeze=False)[:, 0]

    # Shade the rpm bands FINDINGS flagged on the reviewed pull.
    m = REVIEW_RE.search(block or "")
    reviewed = m.group(1) if m else "1"
    bands = []
    if by_rpm:
        for sev, n, lo, hi in BAND_RE.findall(block or ""):
            if n == reviewed:
                lo_f = float(lo)
                hi_f = float(hi) if hi else lo_f + 200
                bands.append((lo_f, hi_f, HIGH_BAND if sev == "HIGH" else MED_BAND))

    for ax, kind in zip(axes, panels):
        for lo, hi, colr in bands:
            ax.axvspan(lo, hi, color=colr, alpha=0.13, linewidth=0)
        if kind == "boost":
            _style(ax, "boost (psi)")
            ax.plot(*_clean(x, boost), color=ACTUAL, linewidth=1.8, label=blabel)
            if tgt is not None:
                ax.plot(*_clean(x, tgt), color=TARGET, linewidth=1.4, linestyle="--", label="target")
            if "wg" in roles:
                ax2 = ax.twinx()
                ax2.plot(*_clean(x, _col(rows, roles["wg"], to_float)), color=EXTRA, linewidth=1.0, alpha=0.8, label="wastegate %")
                ax2.set_ylim(0, 105)
                ax2.tick_params(colors=MUTED, labelsize=8)
                ax2.set_ylabel("wastegate %", color=EXTRA, fontsize=9)
                for s in ax2.spines.values():
                    s.set_color(GRID)
        elif kind == "lambda":
            _style(ax, "lambda")
            # Clip fuel cut (lambda 2+) so it does not flatten the part that matters.
            ax.plot(*_clean(x, _col(rows, roles["lambda"], to_float), 0.6, 1.25), color=ACTUAL, linewidth=1.8, label="lambda")
            if "lambda_sp" in roles:
                ax.plot(*_clean(x, _col(rows, roles["lambda_sp"], to_float), 0.6, 1.25), color=TARGET,
                        linewidth=1.4, linestyle="--", label="setpoint")
        elif kind == "timing":
            _style(ax, "timing (°)")
            if "ign" in roles:
                ax.plot(*_clean(x, _col(rows, roles["ign"], to_float)), color=ACTUAL, linewidth=1.8, label="timing")
            if knock_cols:
                ax2 = ax.twinx()
                worst = 0.0
                for k, (cyl, ci) in enumerate(sorted(knock_cols.items())):
                    kv = [abs(v) if v is not None else None for v in _col(rows, ci, to_float)]
                    worst = max([worst] + [v for v in kv if v is not None])
                    ax2.plot(*_clean(x, kv), color=KNOCK[k % len(KNOCK)], linewidth=1.1, label=f"knock cyl {cyl}")
                ax2.set_ylim(0, max(4.0, worst * 1.2))
                ax2.tick_params(colors=MUTED, labelsize=8)
                ax2.set_ylabel("knock retard (°)", color=KNOCK[0], fontsize=9)
                for s in ax2.spines.values():
                    s.set_color(GRID)
                handles, labels = ax2.get_legend_handles_labels()
                if handles:
                    leg = ax2.legend(handles, labels, loc="upper right", fontsize=8, ncol=len(handles),
                                     facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
                    leg.get_frame().set_alpha(0.9)
        elif kind == "rail":
            u = unit("fp_di")
            scale, ulabel = (0.001, "MPa") if u == "kpa" else (1.0, u or "")
            _style(ax, f"HPFP rail ({ulabel})")
            ax.plot(*_clean(x, [v * scale if v is not None else None for v in _col(rows, roles["fp_di"], to_float)]),
                    color=ACTUAL, linewidth=1.8, label="rail")
            ax.plot(*_clean(x, [v * scale if v is not None else None for v in _col(rows, roles["fp_di_sp"], to_float)]),
                    color=TARGET, linewidth=1.4, linestyle="--", label="setpoint")
        elif kind == "lowside":
            u = unit("fp_mpi")
            scale, ulabel = (0.01, "bar") if u == "kpa" else (1.0, u or "")
            _style(ax, f"low-side fuel ({ulabel})")
            ax.plot(*_clean(x, [v * scale if v is not None else None for v in _col(rows, roles["fp_mpi"], to_float)]),
                    color=ACTUAL, linewidth=1.8, label="LPFP pressure")
            ax.plot(*_clean(x, [v * scale if v is not None else None for v in _col(rows, roles["fp_mpi_sp"], to_float)]),
                    color=TARGET, linewidth=1.4, linestyle="--", label="setpoint")
            if "lpfp_duty" in roles:
                ax2 = ax.twinx()
                ax2.plot(*_clean(x, _col(rows, roles["lpfp_duty"], to_float)), color=EXTRA, linewidth=1.0,
                         alpha=0.8, label="pump duty %")
                ax2.set_ylim(0, 105)
                ax2.tick_params(colors=MUTED, labelsize=8)
                ax2.set_ylabel("pump duty %", color=EXTRA, fontsize=9)
                for s in ax2.spines.values():
                    s.set_color(GRID)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            leg = ax.legend(handles, labels, loc="upper left", fontsize=8, facecolor=PANEL, edgecolor=GRID,
                            labelcolor=INK, ncol=len(handles))
            leg.get_frame().set_alpha(0.9)

    axes[-1].set_xlabel(xlabel, color=INK, fontsize=10)
    head = f"Pull {reviewed}: {m.group(2)}" if m else "Reviewed pull"
    head = re.sub(r",\s*(?:peak boost|lambda at peak boost|worst knock|peak timing|IAT)\b.*$", "", head)
    fig.suptitle(f"{head}   ·   {filename}", color=INK, fontsize=12, x=0.01, ha="left")
    note = []
    if bands:
        note.append("shaded = where the review found a problem (red high, amber medium)")
    if not by_rpm:
        note.append("rpm went up and down (launch or wheelspin), so this is plotted against time")
    if note:
        fig.text(0.01, 0.005, " · ".join(note), color=MUTED, fontsize=8.5, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    return buf.getvalue()
