"""Parse ASAM A2L files into calibration records.

An A2L is the ECU's own definition file: every tunable characteristic and every
loggable measurement, with a description, address, type and limits. It is already
structured, so unlike the Funktionsrahmen there is nothing to reconstruct - it just
has to be read.

Two things matter for using it alongside the FR:

  * A2L names are lower case, FR names are upper case. Same namespace, different
    case, so anything that cross-references them has to normalise or it will find
    nothing.
  * Each file describes ONE ECU family. Mixing them without a source tag is how a
    bot ends up citing a Macan calibration for a Golf question, so every record
    carries where it came from.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

# /begin CHARACTERISTIC <name> "<description>" <TYPE> <address> ... /end
BLOCK_RE = re.compile(
    r"/begin\s+(CHARACTERISTIC|MEASUREMENT|AXIS_PTS)\s+(.*?)/end\s+\1",
    re.S,
)
QUOTED_RE = re.compile(r'"([^"]*)"')
ADDRESS_RE = re.compile(r"\b(0x[0-9a-fA-F]{4,})\b")
TYPE_RE = re.compile(r"\b(VALUE|CURVE|MAP|CUBOID|ASCII|VAL_BLK|CUBE_4|CUBE_5)\b")
UNIT_RE = re.compile(r'PHYS_UNIT\s+"([^"]*)"')
NUMBER_RE = re.compile(r"^-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?$")

MAX_DESC = 300


def _limits(tokens: list[str]) -> tuple[float | None, float | None]:
    """Lower/upper limit are the last two bare numbers before any sub-block."""
    numbers: list[float] = []
    for token in tokens:
        if NUMBER_RE.match(token):
            try:
                numbers.append(float(token))
            except ValueError:
                pass
    if len(numbers) >= 2:
        return numbers[-2], numbers[-1]
    return None, None


def parse(path: Path, source: str, chunk_mb: int = 64) -> Iterator[dict]:
    """Yield one record per characteristic / measurement / axis in the file.

    Read in overlapping slices so a 68 MB file never lands in memory whole.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    seen: set[tuple[str, str]] = set()
    for match in BLOCK_RE.finditer(text):
        kind, body = match.group(1), match.group(2)
        # Stop at the first nested sub-block; the header fields are all before it.
        head = body.split("/begin", 1)[0]
        tokens = head.split()
        if not tokens:
            continue
        name = tokens[0].strip('"')
        if not name or name.startswith("/"):
            continue
        key = (name.upper(), kind)
        if key in seen:
            continue
        seen.add(key)

        described = QUOTED_RE.search(head)
        description = (described.group(1).strip() if described else "")[:MAX_DESC]
        address = ADDRESS_RE.search(head)
        shape = TYPE_RE.search(head)
        unit = UNIT_RE.search(body)
        lower, upper = _limits(tokens[1:])

        yield {
            "name": name.upper(),
            "raw_name": name,
            "kind": kind,
            "shape": shape.group(1) if shape else "",
            "desc": description,
            "address": address.group(1) if address else "",
            "lower": lower,
            "upper": upper,
            "unit": unit.group(1) if unit else "",
            "source": source,
        }


def embed_text(record: dict) -> str:
    """Description leads - that is what a question is actually about."""
    bits = [record.get("desc") or record["name"], record["name"]]
    shape = record.get("shape") or record.get("kind", "")
    if shape:
        bits.append("map" if shape in {"MAP", "CUBOID", "CUBE_4", "CUBE_5"}
                    else "curve" if shape == "CURVE"
                    else "measurement" if record.get("kind") == "MEASUREMENT"
                    else "value")
    if record.get("unit"):
        bits.append(f"unit {record['unit']}")
    bits.append(record["source"])
    return " | ".join(b for b in bits if b)
