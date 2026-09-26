"""Pull page text out of a Funktionsrahmen PDF into a JSONL checkpoint.

Extraction is the slow, fragile half of building the index, so it is kept apart
from chunking and embedding: run this once and re-chunk as often as you like.

The FR is a Continental/LaTeX document with a very regular shape:

    Copyright (c)Continental AG. ...boilerplate...     <- every single page
    LACO, Lambda adaptation                            <- function label + title
    ...content...

so the label a tuner actually searches by is sitting in the running header, and
we just have to read it rather than infer anything.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

# The same ~40 words of legal boilerplate open every page. Left in, they would
# dominate short pages and make every embedding look alike.
COPYRIGHT_RE = re.compile(
    r"Copyright\s*©?\s*Continental AG\..*?reserved\.\s*", re.S | re.I
)

# "LACO, Lambda adaptation" / "ENLU, Oil pressure". Anchored to the top of the
# page because the same shape appears mid-page inside tables.
HEADER_RE = re.compile(r"\A([A-Z][A-Za-z0-9_]{1,19}),[ \t]+(.{2,90})$", re.M)

# Cross-reference pages are wall-to-wall dot leaders ("def . . . . . . INJR:..").
# They carry no explanation, only pointers, and they retrieve badly - a query
# about wastegates would match a page that merely lists the word.
DOT_LEADER_RE = re.compile(r"\.\s\.\s\.")

# Identifiers: MFF_KGH_ADD_LAM_AD, KFMIRL, NC_CBK_EX_NR. Used for exact-match
# boosting and for the vocabulary the trigger checks against.
IDENT_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+){1,6}\b|\b[A-Z]{2,}[A-Z0-9]{2,}\b")

PAGE_TYPE_CONTENT = "content"
PAGE_TYPE_XREF = "xref"


@dataclass
class Page:
    page: int          # 1-based, matches what a PDF reader shows
    label: str         # e.g. "LACO"
    title: str         # e.g. "Lambda adaptation"
    kind: str          # content | xref
    text: str


def clean_page(raw: str) -> str:
    return COPYRIGHT_RE.sub("", raw or "").strip()


def classify(text: str) -> str:
    """Cross-reference index pages are mostly dot leaders; skip them."""
    if not text:
        return PAGE_TYPE_XREF
    leaders = len(DOT_LEADER_RE.findall(text))
    # One leader per ~2 lines is already an index page, not prose.
    return PAGE_TYPE_XREF if leaders >= max(3, text.count("\n") // 3) else PAGE_TYPE_CONTENT


def identifiers(text: str, limit: int = 40) -> list[str]:
    seen: list[str] = []
    for token in IDENT_RE.findall(text or ""):
        if token not in seen:
            seen.append(token)
            if len(seen) >= limit:
                break
    return seen


def extract(pdf_path: Path, out_path: Path, *, progress_every: int = 2000) -> dict:
    doc = pymupdf.open(pdf_path)
    if doc.needs_pass and not doc.authenticate(""):
        raise RuntimeError(
            f"{pdf_path.name} needs a password that we do not have - cannot extract."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"pages": doc.page_count, "content": 0, "xref": 0, "labelled": 0, "carried": 0}
    last_label = ""
    last_title = ""

    with out_path.open("w", encoding="utf-8") as fh:
        for index in range(doc.page_count):
            text = clean_page(doc[index].get_text("text"))
            match = HEADER_RE.match(text)
            if match:
                last_label, last_title = match.group(1), match.group(2).strip()
                stats["labelled"] += 1
            elif last_label:
                # Continuation page of the same function - inherit the heading.
                stats["carried"] += 1
            kind = classify(text)
            stats["content" if kind == PAGE_TYPE_CONTENT else "xref"] += 1
            page = Page(
                page=index + 1,
                label=last_label,
                title=last_title,
                kind=kind,
                text=text,
            )
            fh.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")
            if progress_every and (index + 1) % progress_every == 0:
                print(f"  ...{index + 1}/{doc.page_count} pages", flush=True)
    doc.close()
    return stats


def load_pages(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract FR PDF pages to JSONL.")
    parser.add_argument("pdf")
    parser.add_argument("--out", default="data/fr_pages.jsonl")
    args = parser.parse_args()

    result = extract(Path(args.pdf), Path(args.out))
    print(json.dumps(result, indent=2))
