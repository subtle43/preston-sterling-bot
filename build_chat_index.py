"""Phase 2: turn raw.jsonl into a searchable index. Reads no network.

Chunking is the whole game here, and chat is not the Funktionsrahmen. FR pages are
structured technical prose that embeds well in 250-word slabs. A Discord message
averages 59 characters - "yeah same", "what psi" - and a vector for one of those
means nothing. So messages are grouped into CONVERSATIONS: consecutive messages in
one channel with no long silence between them, packed to roughly the same 250
words the FR index uses, with the speaker's name kept inline so a question like
"what did member_a say about his turbo" has a name to match on.

Safe to re-run at any time, including while the crawl is still going - it only
reads raw.jsonl and rebuilds from whatever is there. That is the whole reason
scraping and indexing are separate programs.

    python build_chat_index.py            # build from raw.jsonl
    python build_chat_index.py --dry-run  # chunk and report, embed nothing
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ollama import Client

from config import Settings

EMBED_MODEL = "bge-m3"
BATCH = 64                  # measured ~44 chunks/sec on this box at this size

# Same slab size as the FR index, for the same reason: big enough to carry an
# argument, small enough that one topic dominates the vector.
CHUNK_WORDS = 250
# A silence this long ends the conversation. Two people talking at 14:00 and one
# person posting at 19:00 are not the same exchange and must not share a vector.
GAP_SECONDS = 15 * 60
# Carried into the next chunk so a point made either side of a boundary is still
# findable. The FR index overlaps by words; here whole messages are the unit,
# because half a message is not a thing anybody said.
OVERLAP_MESSAGES = 2
# A chunk with less than this is a fragment - a lone "any ideas?" that happened to
# sit after a long gap. Not worth a row in the matrix.
MIN_CHUNK_WORDS = 25


def load_raw(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a partially-written last line while the crawl runs
    return rows


def build_chunks(rows: list[dict]) -> list[dict]:
    """Group messages into conversation windows, one channel at a time."""
    by_channel: dict[int, list[dict]] = {}
    for r in rows:
        by_channel.setdefault(r["channel_id"], []).append(r)

    chunks: list[dict] = []
    for channel_id, msgs in by_channel.items():
        msgs.sort(key=lambda m: m["ts"])
        window: list[dict] = []
        words = 0

        def flush() -> None:
            nonlocal window, words
            if words >= MIN_CHUNK_WORDS and window:
                chunks.append(_make_chunk(window))
            # Carry the tail so a thought spanning the boundary stays findable.
            window = window[-OVERLAP_MESSAGES:] if len(window) > OVERLAP_MESSAGES else []
            words = sum(len(m["text"].split()) for m in window)

        for msg in msgs:
            if window and msg["ts"] - window[-1]["ts"] > GAP_SECONDS:
                flush()
                window, words = [], 0        # a silence breaks the thread outright
            window.append(msg)
            words += len(msg["text"].split())
            if words >= CHUNK_WORDS:
                flush()
        if words >= MIN_CHUNK_WORDS and window:
            chunks.append(_make_chunk(window))
    return chunks


def _make_chunk(window: list[dict]) -> dict:
    speakers: list[str] = []
    for m in window:
        if m["author"] not in speakers:
            speakers.append(m["author"])
    return {
        "channel": window[0]["channel"],
        "channel_id": window[0]["channel_id"],
        "ts_start": window[0]["ts"],
        "ts_end": window[-1]["ts"],
        "speakers": speakers,
        "author_ids": sorted({m["author_id"] for m in window}),
        "text": "\n".join(f"{m['author']}: {m['text']}" for m in window),
    }


def embed_text(chunk: dict) -> str:
    """What actually gets embedded.

    The channel name and the speakers are prepended deliberately: retrieval has to
    answer "what did X say about Y", and a name only matches if it is in the text
    that became the vector.
    """
    when = time.strftime("%Y-%m", time.localtime(chunk["ts_start"]))
    return (f"#{chunk['channel']} ({when}) - {', '.join(chunk['speakers'])}\n"
            f"{chunk['text']}")


def embed_all(chunks: list[dict], host: str) -> np.ndarray:
    client = Client(host=host)
    out = np.zeros((len(chunks), 1024), dtype=np.float32)
    t0 = time.monotonic()
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        resp = client.embed(
            model=EMBED_MODEL,
            input=[embed_text(c) for c in batch],
            keep_alive="5m",
        )
        vecs = np.asarray(resp.embeddings, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0          # normalise once, so search is a dot product
        out[i:i + len(batch)] = vecs / norms
        done = i + len(batch)
        rate = done / max(time.monotonic() - t0, 0.01)
        eta = (len(chunks) - done) / max(rate, 0.01)
        print(f"  embedded {done:,}/{len(chunks):,}  {rate:,.0f}/s  "
              f"eta {eta / 60:.1f} min", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="chunk and report without embedding")
    args = ap.parse_args()

    settings = Settings.load()
    out_dir = Path(settings.chat_index_dir)
    raw_path = out_dir / "raw.jsonl"
    if not raw_path.exists():
        raise SystemExit(f"{raw_path} does not exist - run scrape_chat.py first.")

    rows = load_raw(raw_path)
    print(f"messages : {len(rows):,}")
    chunks = build_chunks(rows)
    if not chunks:
        raise SystemExit("Nothing to index yet.")
    words = [len(c["text"].split()) for c in chunks]
    print(f"chunks   : {len(chunks):,}")
    print(f"words/chunk: mean {sum(words) // len(words)}, "
          f"min {min(words)}, max {max(words)}")
    print(f"channels : {len({c['channel_id'] for c in chunks})}")
    print(f"speakers : {len({s for c in chunks for s in c['speakers']})}")
    if args.dry_run:
        print("\n--- sample chunk ---")
        print(embed_text(chunks[len(chunks) // 2])[:700])
        return

    print(f"\nembedding with {EMBED_MODEL}...")
    vectors = embed_all(chunks, settings.ollama_host)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "vectors.npy", vectors)
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    # author_id -> the name they are indexed under. Display names change; ids do
    # not. Without this, a member who renames themselves loses their entire
    # history as far as retrieval is concerned - member_f renamed to member_h and
    # his 14,000 messages were unreachable under the new name until this existed.
    #
    # Every name they posted under is kept too, and the bot tops the list up with
    # usernames and current nicks from the guild (ChatIndex.refresh_names), so
    # those survive a rebuild rather than being refetched. Same key layout:
    # {id: {"indexed": name, "names": [every name]}}.
    prior: dict[str, list[str]] = {}
    checked: dict[str, float] = {}
    joined: dict[str, float] = {}
    speakers_path = out_dir / "speakers.json"
    if speakers_path.exists():
        try:
            for aid, entry in json.loads(speakers_path.read_text(encoding="utf-8")).items():
                if isinstance(entry, dict):
                    prior[aid] = [str(n) for n in entry.get("names") or []]
                    checked[aid] = float(entry.get("checked") or 0.0)
                    joined[aid] = float(entry.get("joined") or 0.0)
        except Exception:
            pass
    counts: dict[str, dict[str, int]] = {}
    for r in rows:
        seen = counts.setdefault(str(r["author_id"]), {})
        seen[r["author"]] = seen.get(r["author"], 0) + 1
    by_id: dict[str, dict] = {}
    for aid, seen in counts.items():
        indexed = max(seen, key=seen.get)
        names: list[str] = [indexed]
        for n in [*seen, *prior.get(aid, [])]:
            if n and n.lower() not in {x.lower() for x in names}:
                names.append(n)
        by_id[aid] = {
            "indexed": indexed, "names": names,
            "checked": checked.get(aid, 0.0), "joined": joined.get(aid, 0.0),
        }
    speakers_path.write_text(
        json.dumps(by_id, ensure_ascii=False, indent=0), encoding="utf-8"
    )
    print(f"speakers.json: {len(by_id):,} author ids")

    (out_dir / "meta.json").write_text(json.dumps({
        "model": EMBED_MODEL,
        "dims": int(vectors.shape[1]),
        "chunks": len(chunks),
        "messages": len(rows),
        "chunk_words": CHUNK_WORDS,
        "gap_seconds": GAP_SECONDS,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=1), encoding="utf-8")
    size = (out_dir / "vectors.npy").stat().st_size / 1e6
    print(f"\nwrote {len(chunks):,} chunks, vectors.npy {size:,.1f} MB -> {out_dir}")


if __name__ == "__main__":
    main()
