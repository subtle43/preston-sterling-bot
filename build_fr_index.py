"""Turn extracted FR pages into an embedded, searchable index.

Run frextract.py first, then this. Splitting the two means a chunking tweak
costs six minutes instead of re-reading a 245 MB PDF.

Output lands in data/fr_index/:
    chunks.jsonl  text + {label, title, pages, identifiers}
    vectors.npy   float32 [n, 1024], already L2-normalised by bge-m3
    vocab.json    every identifier seen, for exact-match triggering
    meta.json     model, dims, counts - checked at load time
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from ollama import Client

from frextract import PAGE_TYPE_CONTENT, identifiers, load_pages

CHUNK_WORDS = 250
OVERLAP_WORDS = 60
BATCH = 64
EMBED_MODEL = "bge-m3"
# bge-m3 handles 8192 tokens; a 250-word chunk is nowhere near it, but a runaway
# table page could be, and a silently truncated embedding is worse than a short one.
MAX_CHARS = 6000


def build_chunks(pages_path: Path) -> list[dict]:
    """Chunk within runs of consecutive pages that share a function label.

    Chunking across a function boundary would blend two unrelated systems into
    one embedding, which is the classic way RAG starts citing the wrong table.
    """
    chunks: list[dict] = []
    run: list[tuple[str, int]] = []   # (word, page)
    run_label = run_title = ""

    def flush() -> None:
        nonlocal run
        if not run:
            return
        step = max(1, CHUNK_WORDS - OVERLAP_WORDS)
        for start in range(0, len(run), step):
            window = run[start : start + CHUNK_WORDS]
            if not window:
                break
            # A trailing sliver that the previous window already covered.
            if start and len(window) <= OVERLAP_WORDS:
                break
            body = " ".join(w for w, _ in window)[:MAX_CHARS]
            first, last = window[0][1], window[-1][1]
            chunks.append(
                {
                    "label": run_label,
                    "title": run_title,
                    "page_start": first,
                    "page_end": last,
                    "identifiers": identifiers(body, limit=25),
                    "text": body,
                }
            )
        run = []

    for page in load_pages(pages_path):
        if page["kind"] != PAGE_TYPE_CONTENT:
            continue
        text = (page.get("text") or "").strip()
        if not text:
            continue
        label = page.get("label") or ""
        if label != run_label:
            flush()
            run_label, run_title = label, page.get("title") or ""
        run.extend((w, page["page"]) for w in text.split())
    flush()
    return chunks


def embed_text(chunk: dict) -> str:
    """What actually gets embedded - the heading matters as much as the body.

    'LACO Lambda adaptation' in the vector is what makes "what is LACO" work
    without a separate keyword index.
    """
    head = f"{chunk['label']} {chunk['title']}".strip()
    return f"{head}\n{chunk['text']}" if head else chunk["text"]


def embed_all(chunks: list[dict], host: str) -> np.ndarray:
    client = Client(host=host)
    out = np.zeros((len(chunks), 1024), dtype=np.float32)
    started = time.time()
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        resp = client.embed(model=EMBED_MODEL, input=[embed_text(c) for c in batch])
        out[i : i + len(batch)] = np.asarray(resp.embeddings, dtype=np.float32)
        done = i + len(batch)
        if done % (BATCH * 20) == 0 or done == len(chunks):
            rate = done / max(1e-6, time.time() - started)
            eta = (len(chunks) - done) / max(1e-6, rate)
            print(f"  embedded {done}/{len(chunks)}  {rate:.0f}/s  eta {eta/60:.1f}m", flush=True)
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build the FR vector index.")
    parser.add_argument("--pages", default="data/fr_pages.jsonl")
    parser.add_argument("--out", default="data/fr_index")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Chunking...")
    chunks = build_chunks(Path(args.pages))
    if not chunks:
        raise SystemExit("No content chunks produced - was frextract.py run?")
    words = sum(len(c["text"].split()) for c in chunks)
    print(f"  {len(chunks)} chunks, {words/1e6:.2f}M words")

    vocab = sorted({i for c in chunks for i in c["identifiers"]})
    labels = sorted({c["label"] for c in chunks if c["label"]})
    print(f"  {len(vocab)} identifiers, {len(labels)} function labels")

    print(f"Embedding with {EMBED_MODEL}...")
    vectors = embed_all(chunks, args.host)

    norms = np.linalg.norm(vectors, axis=1)
    print(f"  norms min={norms.min():.4f} max={norms.max():.4f}  nan={np.isnan(vectors).any()}")

    np.save(out_dir / "vectors.npy", vectors)
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    (out_dir / "vocab.json").write_text(
        json.dumps({"identifiers": vocab, "labels": labels}, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "model": EMBED_MODEL,
                "dims": int(vectors.shape[1]),
                "chunks": len(chunks),
                "chunk_words": CHUNK_WORDS,
                "overlap_words": OVERLAP_WORDS,
                "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out_dir}/  ({vectors.nbytes/1e6:.0f} MB of vectors)")


if __name__ == "__main__":
    main()
