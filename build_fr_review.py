"""Sample real tuning questions from the chat archive and record what the FR
search returns for each, for a human review sheet.

eval_fr.py is saturated (37/37), so it can no longer show whether a retrieval
change helps. This builds the next test set from questions members actually
asked: a reviewer marks each result right / partly / wrong and names the
correct function where they know it, and those marks become the new cases.

    .venv\\Scripts\\python build_fr_review.py            # -> data/fr_review/questions.json
    .venv\\Scripts\\python build_fr_review.py --n 120
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections import Counter
from pathlib import Path

import frsearch
from config import Settings

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "chat_index" / "raw.jsonl"
EXCLUDE = ROOT / "data" / "train" / "exclude.txt"
OUT = ROOT / "data" / "fr_review" / "questions.json"
# Not the Simos 18.10 engine ECU: gearbox, other platforms, flashing tools.
SKIP_CHANNEL_RE = re.compile(r"dsg|tcu|non-mqb|simos-?19|flashing|haldex|marketplace", re.I)
MENTION_RE = re.compile(r"<@!?\d+>")
PER_FUNCTION = 3          # at most this many questions whose top hit is the same function
SEED = 20260924


def candidates() -> list[dict]:
    excluded = set()
    if EXCLUDE.exists():
        excluded = {ln.strip() for ln in EXCLUDE.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.startswith("#")}
    seen: set[str] = set()
    out = []
    with RAW.open(encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            text = " ".join(MENTION_RE.sub("@member", m["text"] or "").split())
            words = len(text.split())
            if not (8 <= words <= 60) or "?" not in text or "http" in text:
                continue
            if str(m["author_id"]) in excluded or SKIP_CHANNEL_RE.search(m["channel"]):
                continue
            # A real subject, not just a question shape.
            if not (frsearch.asks_for_fr(text) and frsearch.ECU_TERM_RE.search(text)):
                continue
            key = text.lower()[:50]
            if key in seen:
                continue
            seen.add(key)
            out.append({"msg_id": str(m["id"]), "q": text, "channel": m["channel"], "ts": m["ts"]})
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    args = ap.parse_args()

    settings = Settings.load()
    index = frsearch.FRIndex(settings.fr_index_dir, settings.ollama_host)
    pool = candidates()
    rng = random.Random(SEED)
    # The main tuning channel first, the rest shuffled in behind it.
    rng.shuffle(pool)
    pool.sort(key=lambda c: c["channel"] != "ecu-tuning")
    print(f"{len(pool)} candidate questions")

    picked: list[dict] = []
    per_fn: Counter = Counter()
    for cand in pool[: args.n * 4]:
        q = cand["q"]
        # The same budget the bot uses: full top_k only when a label is named.
        top_k = settings.fr_top_k if index.known_identifiers(q) else max(4, settings.fr_top_k // 4)
        prose = await index.search(q, top_k=top_k)
        params = await index.search_parameters(q, top_k=8)
        functions, seen_fn = [], set()
        for score, chunk in prose:
            if chunk["label"] in seen_fn:
                continue
            seen_fn.add(chunk["label"])
            functions.append({
                "label": chunk["label"], "title": chunk["title"], "score": round(score, 3),
                "pages": f"{chunk['page_start']}-{chunk['page_end']}",
                "snippet": " ".join(frsearch.clean_excerpt(chunk["text"]).split()[:45]),
            })
        top = functions[0]["label"] if functions else ""
        if per_fn[top] >= PER_FUNCTION:
            continue
        per_fn[top] += 1
        picked.append({
            **cand,
            "id": len(picked) + 1,
            "best": round(max([s for s, _ in prose] + [s for s, _ in params] + [0.0]), 3),
            "functions": functions[:4],
            "params": [
                {"name": r["name"], "desc": r.get("desc") or "", "label": r.get("label") or "",
                 "page": r.get("page"), "score": round(s, 3)}
                for s, r in params[:6]
            ],
        })
        if len(picked) >= args.n:
            break

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(picked, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(picked)} questions -> {OUT}")
    print("channels:", Counter(p["channel"] for p in picked).most_common(8))
    print("top functions:", len(per_fn), "distinct")


if __name__ == "__main__":
    asyncio.run(main())
