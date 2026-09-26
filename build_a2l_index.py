"""Embed one or more A2L files into a searchable calibration index.

    python build_a2l_index.py "path\\SCGA0531_C_OEM.a2l=SCGA0531" \\
                              "path\\SC8S5031_C_OEM.a2l=SC8S5031"

Each record keeps its source ECU. Two A2Ls describe two different engines, and a
calibration from the wrong one is worse than no answer, so the tag travels with
every record and is shown on every line the model sees.

Outputs into data/fr_index/:
    a2l.jsonl      one record per characteristic / measurement / axis
    a2l.npy        float32 [n, 1024]
    a2l_meta.json  models, counts, and which sources are in here
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from ollama import Client

import a2lparse

BATCH = 96           # short records, but every one costs a socket
PAUSE_SECONDS = 3.0  # between checkpoints, so TIME_WAIT can drain
EMBED_MODEL = "bge-m3"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build the A2L calibration index.")
    parser.add_argument("files", nargs="+", help="path=SOURCETAG pairs")
    parser.add_argument("--out", default="data/fr_index")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    sources: list[str] = []
    for spec in args.files:
        path_str, _, tag = spec.partition("=")
        path = Path(path_str)
        if not path.exists():
            raise SystemExit(f"No such A2L: {path}")
        tag = tag or path.stem
        sources.append(tag)
        started = time.time()
        found = [r for r in a2lparse.parse(path, tag) if r["desc"]]
        records.extend(found)
        print(f"  {tag}: {len(found):,} described records "
              f"({time.time() - started:.0f}s)", flush=True)

    if not records:
        raise SystemExit("Nothing to index.")
    print(f"total {len(records):,} records from {len(sources)} ECU(s)")

    client = Client(host=args.host)
    # np.save appends ".npy" unless the name already ends in it, so a path of
    # "a2l.npy.part" is written as "a2l.npy.part.npy" and the resume check for
    # the original name never matches. Name it so the two agree.
    part_path = out / "a2l_part.npy"
    legacy_part = out / "a2l.npy.part.npy"
    if legacy_part.exists() and not part_path.exists():
        legacy_part.replace(part_path)
    progress_path = out / "a2l.progress"

    vectors = np.zeros((len(records), 1024), dtype=np.float32)
    start_at = 0
    # Resume: 188k records is ~35 minutes, and losing that to a transient socket
    # error once was enough.
    if part_path.exists() and progress_path.exists():
        try:
            saved = np.load(part_path)
            done_n = int(progress_path.read_text().strip())
            if saved.shape == vectors.shape and 0 < done_n <= len(records):
                vectors, start_at = saved, done_n
                print(f"resuming from {start_at:,}", flush=True)
        except Exception:
            print("checkpoint unreadable, starting over", flush=True)

    started = time.time()
    for i in range(start_at, len(records), BATCH):
        batch = records[i : i + BATCH]
        payload = [a2lparse.embed_text(r) for r in batch]
        for attempt in range(6):
            try:
                resp = client.embed(model=EMBED_MODEL, input=payload)
                break
            except Exception as exc:
                # Windows runs out of ephemeral ports long before it runs out of
                # anything else: tens of thousands of sockets pile up in TIME_WAIT
                # and the next connect fails. Waiting lets them drain.
                if attempt == 5:
                    np.save(part_path, vectors)
                    progress_path.write_text(str(i))
                    raise SystemExit(
                        f"Giving up at {i:,} after 6 tries: {exc}\n"
                        f"Progress saved - rerun the same command to resume."
                    )
                wait = 20 * (attempt + 1)
                print(f"  socket/API error at {i:,}, waiting {wait}s "
                      f"(attempt {attempt + 1}/6): {str(exc)[:90]}", flush=True)
                time.sleep(wait)
        vectors[i : i + len(batch)] = np.asarray(resp.embeddings, dtype=np.float32)
        done = i + len(batch)
        if done % (BATCH * 20) == 0 or done == len(records):
            rate = (done - start_at) / max(1e-6, time.time() - started)
            print(f"  {done:,}/{len(records):,}  {rate:.0f}/s  "
                  f"eta {(len(records)-done)/max(1e-6,rate)/60:.0f}m", flush=True)
            np.save(part_path, vectors)
            progress_path.write_text(str(done))
            # Let the socket table drain. Costs a few minutes over the whole run
            # and is the difference between finishing and dying at 75%.
            time.sleep(PAUSE_SECONDS)

    norms = np.linalg.norm(vectors, axis=1)
    print(f"  norms min={norms.min():.4f} max={norms.max():.4f} "
          f"nan={bool(np.isnan(vectors).any())}")

    np.save(out / "a2l.npy", vectors)
    with (out / "a2l.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "a2l_meta.json").write_text(
        json.dumps(
            {
                "model": EMBED_MODEL,
                "dims": int(vectors.shape[1]),
                "records": len(records),
                "sources": sources,
                "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for temp in (part_path, progress_path):
        temp.unlink(missing_ok=True)
    print(f"Wrote {out}/  ({vectors.nbytes/1e6:.0f} MB)")


if __name__ == "__main__":
    sys.exit(main())
