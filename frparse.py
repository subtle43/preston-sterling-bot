"""Reconstruct the Funktionsrahmen's underlying database from its rendered pages.

The FR is not prose. It is a generated reference document: a symbol table plus
thousands of data-definition tables, laid out by LaTeX and printed to PDF. Indexing
it as prose - 250-word chunks, blind - destroys that structure and buries each
parameter's one descriptive sentence in a page of hex ranges, which is why
IP_T_MIN_PU_CS could not be found by any amount of query rewriting.

Two things are recovered here:

  symbols()    from the cross-reference pages: name -> the function that DEFINES it,
               plus the functions that use it. ~43k names, near-perfect exact lookup.
  parameters() from the data-definition tables: one record per parameter, with its
               description standing alone instead of drowning in numeric columns.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Iterator

# A calibration identifier: uppercase runs joined by underscores. The optional
# trailing [..] is a map dimension, e.g. IP_T_MIN_PU_CS[NC_NR_TRANS_TYP_MT].
NAME_RE = re.compile(r"^([A-Z][A-Z0-9]*(?:_[A-Z0-9]+){1,10})\s*(\[[^\]]{1,60}\])?$")

# Column 2 of a data-definition row. "V" = value, "O/V" = output/value, etc.
MODE_RE = re.compile(r"^(?:V|O/V|O|M|C|K|-)$")

# "0... 1H", "8000... 7FFFH", "-1024... 1023.96875"
RANGE_RE = re.compile(r"^[-+0-9A-Fa-f. ]{1,40}H?$")

# Cross-reference lines: "def . . . . . ENOS:Basic_operating_states_AI"
XREF_RE = re.compile(r"^(def|use)\s*[.\s]*\s*([A-Za-z0-9]{2,8}):(\S+)\s*$")

# A description is prose: it has lower-case words and is not itself an identifier.
PROSE_RE = re.compile(r"[a-z]{3}")

# Page furniture that terminates a table.
STOP_LINES = {
    "· Application Conditions", "Application Conditions", "Initialisation:",
    "Activation:", "Deactivation:", "Recurrence:", "File:", "Project:",
    "Document key:", "Baseline:", "Mode", "Coded Limits", "Display Limits",
    "Resolution", "Unit", "Name",
}

MAX_DESC_CHARS = 400


def _lines(text: str) -> list[str]:
    return [line.strip() for line in (text or "").split("\n") if line.strip()]


def symbols(pages: Iterable[dict]) -> dict[str, dict[str, list[str]]]:
    """Parse the cross-reference pages into a symbol table.

    These are the pages the first index threw away as noise. They are in fact a
    complete "where is this defined" listing for the whole document.
    """
    table: dict[str, dict[str, list[str]]] = {}
    current = ""
    for page in pages:
        if page.get("kind") != "xref":
            continue
        for line in _lines(page.get("text") or ""):
            match = NAME_RE.match(line)
            if match:
                current = match.group(1)
                table.setdefault(current, {"def": [], "use": []})
                continue
            ref = XREF_RE.match(line)
            if ref and current:
                entry = f"{ref.group(2)}:{ref.group(3)}"
                bucket = table[current][ref.group(1)]
                if entry not in bucket:
                    bucket.append(entry)
    return table


def _looks_like_axis(rows: list[str], i: int) -> bool:
    """A map's axis row: NAME, a small integer, then two ranges.

    Scalars are name/mode/range/range/resolution/unit/description. Maps insert one
    of these blocks per axis before the description, which is exactly why a fixed
    seven-line reader missed every map - IP_T_MIN_PU_CS among them.
    """
    if i + 4 >= len(rows):
        return False
    return bool(
        NAME_RE.match(rows[i])
        and re.fullmatch(r"\d{1,3}", rows[i + 1])
        and RANGE_RE.match(rows[i + 2])
    )


def parameters(pages: Iterable[dict]) -> Iterator[dict[str, Any]]:
    """Yield one record per calibration parameter found in a data-definition table."""
    for page in pages:
        if page.get("kind") != "content":
            continue
        rows = _lines(page.get("text") or "")
        label = page.get("label") or ""
        title = page.get("title") or ""
        number = page.get("page")
        i = 0
        while i < len(rows) - 5:
            head = NAME_RE.match(rows[i])
            if not head or not MODE_RE.match(rows[i + 1]):
                i += 1
                continue
            name, dimension = head.group(1), (head.group(2) or "")
            unit = rows[i + 5] if i + 5 < len(rows) else ""
            cursor = i + 6
            # Step over any axis blocks that sit between the columns and the text.
            axes: list[str] = []
            while cursor < len(rows) and _looks_like_axis(rows, cursor):
                axis = NAME_RE.match(rows[cursor])
                if axis:
                    axes.append(axis.group(1))
                cursor += 6
            if cursor >= len(rows):
                break
            desc = rows[cursor]
            if (
                desc in STOP_LINES
                or NAME_RE.match(desc)
                or not PROSE_RE.search(desc)
                or len(desc) < 8
            ):
                i += 1
                continue
            yield {
                "name": name,
                "dimension": dimension.strip("[]"),
                "unit": unit if unit not in STOP_LINES else "",
                "axes": axes,
                "desc": desc[:MAX_DESC_CHARS],
                "label": label,
                "title": title,
                "page": number,
            }
            i = cursor + 1


def embed_text(record: dict[str, Any]) -> str:
    """What a parameter record looks like as a retrieval unit.

    The description carries the meaning, so it leads. The name follows because
    people search by name too, and the function gives it context.
    """
    bits = [record["desc"], record["name"]]
    if record.get("unit") and record["unit"] not in {"-", ""}:
        bits.append(f"unit {record['unit']}")
    if record.get("label"):
        bits.append(f"{record['label']} {record.get('title') or ''}".strip())
    return " | ".join(b for b in bits if b)
