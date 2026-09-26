"""Build the symbol table and the parameter dictionary.

Runs off data/fr_pages.jsonl, so the 245 MB PDF is never touched and the existing
prose index (vectors.npy / chunks.jsonl) is left alone - only the two new indexes
are embedded here.

Outputs into data/fr_index/:
    symbols.json      name -> {def: [...], use: [...]}   ~43k entries
    params.jsonl      one record per calibration parameter, with its description
    params.npy        float32 [n, 1024] embeddings of those descriptions
    params_meta.json  model, dims, counts
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from ollama import Client

import frparse
from frextract import load_pages

BATCH = 128           # short records, so a bigger batch than the prose index uses
EMBED_MODEL = "bge-m3"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build FR symbol + parameter indexes.")
    parser.add_argument("--pages", default="data/fr_pages.jsonl")
    parser.add_argument("--out", default="data/fr_index")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Reading pages...")
    pages = list(load_pages(Path(args.pages)))

    print("Parsing the cross-reference symbol table...")
    symbols = frparse.symbols(pages)
    defined = sum(1 for v in symbols.values() if v["def"])
    print(f"  {len(symbols)} symbols, {defined} with a defining function")
    (out / "symbols.json").write_text(
        json.dumps(symbols, ensure_ascii=False), encoding="utf-8"
    )

    print("Parsing parameter tables...")
    params = list(frparse.parameters(pages))
    # One record per name: the first definition wins, later pages repeat it.
    seen: dict[str, dict] = {}
    for record in params:
        seen.setdefault(record["name"], record)
    records = list(seen.values())
    print(f"  {len(params)} rows -> {len(records)} distinct parameters "
          f"({sum(1 for r in records if r['axes'])} maps)")

    print(f"Embedding {len(records)} descriptions with {EMBED_MODEL}...")
    client = Client(host=args.host)
    vectors = np.zeros((len(records), 1024), dtype=np.float32)
    started = time.time()
    for i in range(0, len(records), BATCH):
        batch = records[i : i + BATCH]
        resp = client.embed(
            model=EMBED_MODEL, input=[frparse.embed_text(r) for r in batch]
        )
        vectors[i : i + len(batch)] = np.asarray(resp.embeddings, dtype=np.float32)
        done = i + len(batch)
        if done % (BATCH * 10) == 0 or done == len(records):
            rate = done / max(1e-6, time.time() - started)
            print(f"  {done}/{len(records)}  {rate:.0f}/s  "
                  f"eta {(len(records)-done)/max(1e-6,rate)/60:.1f}m", flush=True)

    norms = np.linalg.norm(vectors, axis=1)
    print(f"  norms min={norms.min():.4f} max={norms.max():.4f} "
          f"nan={bool(np.isnan(vectors).any())}")

    np.save(out / "params.npy", vectors)
    with (out / "params.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "params_meta.json").write_text(
        json.dumps(
            {
                "model": EMBED_MODEL,
                "dims": int(vectors.shape[1]),
                "parameters": len(records),
                "symbols": len(symbols),
                "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out}/  ({vectors.nbytes/1e6:.0f} MB of parameter vectors)")


if __name__ == "__main__":
    main()
