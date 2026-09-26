"""Pictures for the gag commands: /race time slips, /tierlist boards and the
weekly awards card. All arithmetic and drawing, no model calls - the model
writes the words, this makes them look official.

Object API + Agg canvas only (no pyplot): these run in worker threads.
"""

from __future__ import annotations

import io
import random
import textwrap

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle

# -- /race ---------------------------------------------------------------------

BASE_60FT = {"awd": 1.62, "rwd": 1.85, "fwd": 2.02}


def run_quarter(racer: dict, rng: random.Random) -> dict:
    """A believable 1/4-mile pass from power, weight and drivetrain.

    Hale's formulas (ET from weight/power, trap from power/weight) give the
    pass; the 60 ft depends on the drivetrain and how much power a FWD car is
    trying to put through the front tyres; the mishap the model picked adds
    its seconds - or ends the run.
    """
    whp = max(40.0, min(1500.0, float(racer.get("whp") or 200)))
    weight = max(1800.0, min(6500.0, float(racer.get("weight_lb") or 3300)))
    drive = str(racer.get("drivetrain") or "fwd").lower()[:3]
    drive = drive if drive in BASE_60FT else "fwd"
    sixty = BASE_60FT[drive] + rng.uniform(0.0, 0.12)
    if drive == "fwd" and whp > 280:
        sixty += min(0.6, (whp - 280) / 400)          # all smoke, no go
    et = 5.825 * (weight / whp) ** (1 / 3) + (sixty - 1.75) * 0.9
    trap = 234 * (whp / weight) ** (1 / 3)
    lost = max(0.0, min(9.0, float(racer.get("mishap_seconds") or 0)))
    broke = lost >= 5
    et += lost
    trap *= 1 - min(0.35, lost * 0.06)
    rt = rng.uniform(0.42, 0.68)
    return {
        "rt": rt, "60ft": sixty, "330ft": et * 0.395, "eighth": et * 0.64,
        "eighth_mph": trap * 0.80, "1000ft": et * 0.855, "et": et, "mph": trap,
        "broke": broke, "total": rt + et,
    }


def time_slip(left: dict, right: dict, lres: dict, rres: dict, track: str = "PRESTON STERLING DRAGWAY") -> bytes:
    fig = Figure(figsize=(5.2, 5.9), dpi=110, facecolor="#fbfaf4")
    FigureCanvasAgg(fig)
    ink = "#222222"
    mono = {"family": "monospace", "color": ink}
    y = 0.95

    def line(text, size=10, weight="normal", x=0.5, ha="center"):
        nonlocal y
        fig.text(x, y, text, fontsize=size, fontweight=weight, ha=ha, va="top", **mono)
        y -= 0.043 * size / 10

    line(track, 12, "bold")
    line("OFFICIAL TIME SLIP  ·  TRACK PREP: SPILLED MONSTER", 7.5)
    y -= 0.01
    line("-" * 44, 9)
    winner_left = lres["total"] <= rres["total"] and not lres["broke"] or rres["broke"] and not lres["broke"]
    ln = str(left.get("name", "LEFT"))[:16]
    rn = str(right.get("name", "RIGHT"))[:16]
    line(f"{'':<11}{'LEFT':>13}{'RIGHT':>14}", 9, "bold", x=0.19, ha="left")
    line(f"{'Driver':<11}{ln:>13}{rn:>14}", 9, x=0.19, ha="left")
    rows = [("R/T", "rt", "{:.3f}"), ("60'", "60ft", "{:.3f}"), ("330'", "330ft", "{:.3f}"),
            ("1/8", "eighth", "{:.3f}"), ("MPH", "eighth_mph", "{:.2f}"), ("1000'", "1000ft", "{:.3f}"),
            ("1/4", "et", "{:.3f}"), ("MPH", "mph", "{:.2f}")]
    for label, key, fmt in rows:
        lv = "BROKE" if lres["broke"] and key in ("1000ft", "et", "mph") else fmt.format(lres[key])
        rv = "BROKE" if rres["broke"] and key in ("1000ft", "et", "mph") else fmt.format(rres[key])
        line(f"{label:<11}{lv:>13}{rv:>14}", 9, x=0.19, ha="left")
    line("-" * 44, 9)
    line(f"{'':<11}{'** WIN **' if winner_left else '':>13}{'' if winner_left else '** WIN **':>14}", 9, "bold", x=0.19, ha="left")
    margin = abs(lres["total"] - rres["total"])
    if not (lres["broke"] or rres["broke"]):
        line(f"Margin of victory: {margin:.3f} s", 8.5)
    y -= 0.01
    for side, racer in (("L", left), ("R", right)):
        car = str(racer.get("car", ""))[:46]
        line(f"{side}: {car}", 7.5)
        if racer.get("mishap"):
            line(f"   {str(racer['mishap'])[:44]}", 7.5)
    fig.text(0.5, 0.03, "Keep this slip. It is the only proof you ever raced.", fontsize=7,
             ha="center", family="monospace", color="#777777")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    return buf.getvalue()


# -- /tierlist -------------------------------------------------------------------

TIER_COLOURS = {"S": "#ff7f7f", "A": "#ffbf7f", "B": "#ffdf7f", "C": "#ffff7f", "D": "#bfff7f", "F": "#7fbfff"}
TIER_ORDER = ["S", "A", "B", "C", "D", "F"]


def tier_board(topic: str, tiers: dict[str, list[str]]) -> bytes:
    rows = [t for t in TIER_ORDER if t in tiers]
    per_row = 5
    heights = [max(1, -(-len(tiers[t]) // per_row)) for t in rows]
    total = sum(heights)
    fig = Figure(figsize=(10, 1.2 + 0.95 * total), dpi=100, facecolor="#1a1a17")
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.2 + 0.95 * total)
    ax.axis("off")
    top = 1.2 + 0.95 * total
    ax.text(0.2, top - 0.55, textwrap.shorten(topic.upper(), 70), fontsize=17, fontweight="bold",
            color="#f2f2ee", va="center")
    y = top - 1.1
    for t, h in zip(rows, heights):
        band = 0.95 * h
        ax.add_patch(Rectangle((0.05, y - band + 0.03), 1.0, band - 0.06, color=TIER_COLOURS[t]))
        ax.text(0.55, y - band / 2, t, fontsize=30, fontweight="bold", ha="center", va="center", color="#1a1a17")
        ax.add_patch(Rectangle((1.1, y - band + 0.03), 8.85, band - 0.06, color="#2a2a26"))
        for k, name in enumerate(tiers[t]):
            r, c = divmod(k, per_row)
            cx = 1.22 + c * 1.74
            cy = y - 0.47 - r * 0.95
            ax.add_patch(FancyBboxPatch((cx, cy - 0.3), 1.6, 0.6, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        facecolor="#3b3b36", edgecolor=TIER_COLOURS[t], linewidth=1.5))
            # The chart font has no emoji: "member_d🔥" drew a box.
            label = "".join(ch for ch in str(name) if ord(ch) < 0x2000).strip() or str(name)
            ax.text(cx + 0.8, cy, textwrap.shorten(label, 16, placeholder="…"), fontsize=10.5,
                    color="#f2f2ee", ha="center", va="center", fontweight="bold")
        y -= band
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    return buf.getvalue()


# -- weekly awards ---------------------------------------------------------------

GOLD = "#d4a82c"


def awards_card(week: str, awards: list[tuple[str, str, str]]) -> bytes:
    """awards: (award name, winner, one-line blurb)."""
    n = len(awards)
    fig = Figure(figsize=(10, 1.9 + 1.05 * n), dpi=100, facecolor="#141414")
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    h = 1.9 + 1.05 * n
    ax.set_xlim(0, 10)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.text(5, h - 0.55, "THE PRESTON AWARDS", fontsize=26, fontweight="bold", color=GOLD, ha="center", va="center",
            family="serif")
    ax.text(5, h - 1.1, f"for services to disappointment  ·  {week}", fontsize=11, color="#bdbdb4",
            ha="center", va="center", family="serif", style="italic")
    y = h - 1.75
    for award, winner, blurb in awards:
        ax.add_patch(FancyBboxPatch((0.35, y - 0.9), 9.3, 0.82, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    facecolor="#1f1f1d", edgecolor=GOLD, linewidth=1.2))
        ax.text(0.6, y - 0.28, award.upper(), fontsize=12.5, fontweight="bold", color=GOLD, va="center", family="serif")
        ax.text(9.4, y - 0.28, winner, fontsize=13, fontweight="bold", color="#f5f5f0", va="center", ha="right")
        ax.text(0.6, y - 0.64, textwrap.shorten(blurb, 115, placeholder="…"), fontsize=9.5, color="#cfcfc6", va="center")
        y -= 1.05
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    return buf.getvalue()


def _plain(text) -> str:
    """Drop characters the chart fonts cannot draw (emoji)."""
    return "".join(ch for ch in str(text) if ord(ch) < 0x2000).strip()


# -- /stock ------------------------------------------------------------------------

def stock_chart(ticker: str, name: str, points: list[tuple[float, float]], headlines: list[tuple[float, str]],
                rating: str, note: str, unit: str) -> bytes:
    """points: (unix time, value). headlines: (unix time, text), pinned to the line."""
    import datetime as _dt
    from matplotlib.dates import DateFormatter, date2num

    bg, panel, ink, muted = "#0f1115", "#161a21", "#e8eaed", "#8b93a1"
    xs = [_dt.datetime.fromtimestamp(t) for t, _ in points]
    ys = [v for _, v in points]
    up = ys[-1] >= ys[0]
    line = "#26a269" if up else "#e5484d"
    fig = Figure(figsize=(11, 6.6), dpi=100, facecolor=bg)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0.07, 0.14, 0.88, 0.62))
    ax.set_facecolor(panel)
    for s in ax.spines.values():
        s.set_color("#2a2f38")
    ax.grid(True, color="#232833", linewidth=0.8)
    ax.tick_params(colors=muted, labelsize=9)
    ax.plot(xs, ys, color=line, linewidth=2.2, marker="o", markersize=3.5)
    ax.fill_between(xs, ys, min(ys) * 0.9, color=line, alpha=0.12)
    span = (max(ys) - min(ys)) or max(ys) * 0.2 or 1
    ax.set_ylim(min(ys) - span * 0.25, max(ys) + span * 0.9)
    ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
    ax.set_ylabel(unit, color=muted, fontsize=9)
    for k, (t, text) in enumerate(headlines[:5]):
        when = _dt.datetime.fromtimestamp(t)
        j = min(range(len(xs)), key=lambda i: abs(date2num(xs[i]) - date2num(when)))
        yv = ys[j]
        ax.annotate(textwrap.fill(_plain(text), 26), xy=(xs[j], yv),
                    xytext=(xs[j], yv + span * (0.3 + 0.22 * (k % 2))),
                    ha="center", fontsize=8, color=ink,
                    bbox={"boxstyle": "round,pad=0.3", "fc": "#222834", "ec": "#3a4150", "lw": 0.8},
                    arrowprops={"arrowstyle": "->", "color": muted, "lw": 0.9})
    change = (ys[-1] - ys[0]) / ys[0] * 100 if ys[0] else 0
    fig.text(0.07, 0.93, f"${_plain(ticker)}", fontsize=24, fontweight="bold", color=ink)
    fig.text(0.07, 0.885, f"{_plain(name)} Holdings Ltd.  ·  NASDAQ: PAIN", fontsize=10, color=muted)
    fig.text(0.95, 0.93, f"{ys[-1]:.0f} {unit}", fontsize=22, fontweight="bold", color=ink, ha="right")
    fig.text(0.95, 0.885, f"{'+' if up else ''}{change:.1f}% all time", fontsize=11, color=line, ha="right")
    word = (rating or "SELL").upper().split()[0]
    colour = {"BUY": "#26a269", "HOLD": "#d4a82c"}.get(word, "#e5484d")
    fig.text(0.07, 0.8, "ANALYST RATING: " + textwrap.shorten(_plain(rating).upper(), 60, placeholder="..."),
             fontsize=12, fontweight="bold", color=colour)
    fig.text(0.07, 0.035, textwrap.shorten(_plain(note), 150, placeholder="…"), fontsize=9.5, color=muted)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=bg)
    return buf.getvalue()


# -- /factcheck ------------------------------------------------------------------

def fact_card(name: str, pairs: list[tuple[str, str, str, str]], rating: int, headline: str) -> bytes:
    """pairs: (date A, quote A, date B, quote B). rating: 1-4 Pinocchios."""
    n = max(1, len(pairs))
    h = 2.3 + 2.25 * n
    fig = Figure(figsize=(11, h), dpi=100, facecolor="#0b1a33")
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.add_patch(Rectangle((0, h - 1.05), 11, 1.05, color="#c8102e"))
    ax.text(0.35, h - 0.52, "FACT CHECK", fontsize=26, fontweight="bold", color="white", va="center")
    ax.text(10.65, h - 0.52, _plain(name), fontsize=18, fontweight="bold", color="white", va="center", ha="right")
    ax.text(0.35, h - 1.45, textwrap.shorten(_plain(headline).upper(), 90, placeholder="…"), fontsize=12.5,
            fontweight="bold", color="#ffd166", va="center")
    y = h - 1.95
    for da, qa, db, qb in pairs:
        for x, date, quote, tag in ((0.35, da, qa, "THEN"), (5.65, db, qb, "LATER")):
            ax.add_patch(FancyBboxPatch((x, y - 1.85), 5.0, 1.75, boxstyle="round,pad=0.02,rounding_size=0.1",
                                        facecolor="#12264a", edgecolor="#2f4a7a", linewidth=1.2))
            ax.text(x + 0.2, y - 0.3, f"{tag}  ·  {_plain(date)}", fontsize=9.5, color="#8fb3ff",
                    fontweight="bold", va="center")
            ax.text(x + 0.2, y - 0.55, textwrap.fill('"' + _plain(quote)[:230] + '"', 52), fontsize=10,
                    color="white", va="top", style="italic")
        ax.text(5.5, y - 0.95, "vs", fontsize=12, color="#ffd166", ha="center", va="center", fontweight="bold")
        y -= 2.25
    nose = max(1, min(4, int(rating or 1)))
    ax.text(10.65, 0.35, f"RATING: {nose} PINOCCHIO{'S' if nose > 1 else ''}", fontsize=13,
            fontweight="bold", color="#ffd166", ha="right", va="center")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    return buf.getvalue()


# -- /card ---------------------------------------------------------------------------

RARITY = {"common": "#9aa0a6", "rare": "#4f8cff", "epic": "#a855f7", "legendary": "#f5b301", "cursed": "#e5484d"}


def trading_card(name: str, avatar: bytes | None, card: dict) -> bytes:
    import numpy as np
    from PIL import Image

    rarity = str(card.get("rarity") or "common").lower()
    edge = RARITY.get(rarity, RARITY["common"])
    fig = Figure(figsize=(5.2, 7.3), dpi=110, facecolor="#101010")
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 5.2)
    ax.set_ylim(0, 7.3)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.12, 0.12), 4.96, 7.06, boxstyle="round,pad=0,rounding_size=0.25",
                                facecolor=edge, edgecolor="none"))
    ax.add_patch(FancyBboxPatch((0.28, 0.28), 4.64, 6.74, boxstyle="round,pad=0,rounding_size=0.18",
                                facecolor="#f4efe1", edgecolor="none"))
    ax.text(0.45, 6.72, textwrap.shorten(_plain(name), 20, placeholder="…"), fontsize=15, fontweight="bold",
            color="#1a1a1a", va="center")
    try:
        hp = int(float(card.get("hp") or 100))
    except (TypeError, ValueError):
        hp = 100
    ax.text(4.75, 6.72, f"{hp} HP", fontsize=13, fontweight="bold", color="#c8102e", va="center", ha="right")
    ax.text(0.45, 6.42, textwrap.shorten(_plain(card.get("title") or ""), 44, placeholder="…"), fontsize=8.5,
            color="#444", va="center", style="italic")
    ax.add_patch(Rectangle((0.45, 3.95), 4.3, 2.3, facecolor="#2b2b2b", edgecolor="#8a7a4a", linewidth=2))
    if avatar:
        try:
            img = Image.open(io.BytesIO(avatar)).convert("RGB")
            w, hgt = img.size
            target = 4.3 / 2.3
            if w / hgt > target:
                nw = int(hgt * target)
                img = img.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, hgt))
            else:
                nh = int(w / target)
                img = img.crop((0, (hgt - nh) // 2, w, (hgt - nh) // 2 + nh))
            ax.imshow(np.asarray(img), extent=(0.45, 4.75, 3.95, 6.25), aspect="auto", zorder=2)
        except Exception:
            pass
    ax.text(2.6, 3.72, f"{_plain(card.get('type') or 'Normal')} type  ·  {rarity.upper()}", fontsize=8.5,
            color="#333", ha="center", va="center", fontweight="bold")
    y = 3.35
    for move in (card.get("moves") or [])[:2]:
        ax.text(0.5, y, _plain(move.get("name") or "")[:26], fontsize=11.5, fontweight="bold", color="#111",
                va="center")
        ax.text(4.7, y, str(move.get("damage") or "")[:5], fontsize=12, fontweight="bold", color="#111",
                va="center", ha="right")
        ax.text(0.5, y - 0.22, textwrap.fill(_plain(move.get("text") or "")[:120], 58), fontsize=7.6,
                color="#333", va="top")
        y -= 0.95
    ax.plot([0.45, 4.75], [1.4, 1.4], color="#8a7a4a", linewidth=1)
    ax.text(0.5, 1.22, f"weakness: {_plain(card.get('weakness') or '')[:30]}", fontsize=8, color="#333",
            va="center")
    ax.text(4.7, 1.22, f"resistance: {_plain(card.get('resistance') or '')[:22]}", fontsize=8, color="#333",
            va="center", ha="right")
    ax.text(0.5, 0.95, textwrap.fill(_plain(card.get("flavor") or ""), 60)[:260], fontsize=7.5, color="#555",
            va="top", style="italic")
    serial = sum(ord(c) for c in name) % 999
    ax.text(4.75, 0.38, f"#{serial:03d}/999  ·  Preston Sterling TCG", fontsize=6.5, color="#777",
            ha="right", va="center")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    return buf.getvalue()
