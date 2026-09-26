"""Regression suite for Funktionsrahmen retrieval.

Every FR bug so far was found by somebody asking a question in Discord and the
answer being wrong. This turns that into a measurement: a fixed set of questions
with the function (and sometimes the exact parameter) that should come back, so a
retrieval change can be checked before it ships instead of after.

    python eval_fr.py            # recall + false positives
    python eval_fr.py --verbose  # show what came back for each question
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import frsearch
from config import Settings

# (question, acceptable function labels, exact parameter that should appear or "")
CASES: list[tuple[str, set[str], str]] = [
    # --- the ones that failed in real use, kept as fixtures -----------------
    ("What is the ecu address in A05 for OBD readiness", {"OBDC", "DGNC"}, "STATE_READY_OBD"),
    ("can you lookup the best way to tune ignition timing on simos",
     {"IGSP", "IGRE", "KNCK"}, "IGA_KNK_BAS"),
    ("what tables can a use to turn off cat heating", {"EXTC", "EXTD"}, ""),
    ("what tables can I turn off to make my emissions readiness monitors always show good",
     {"OBDC", "DGNC", "ERRM"}, ""),
    ("On a manual transmission car, my throttle is staying open across a shift at "
     "high load, leading to undesired rev hang. What could be wrong?",
     {"ENOS", "TQDR", "TQSP"}, "IP_T_MIN_PU_CS"),
    ("Can you research what i would need to change to be able to turn on impulse combustion?",
     {"ENOS", "IGSP", "IGRE", "EXTC"}, ""),
    ("what tables are there to tune for wastegate", {"CHRG", "CHRA"}, ""),
    # --- exact-name lookups -------------------------------------------------
    ("what is LACO", {"LACO"}, ""),
    ("what does IP_T_MIN_PU_CS do", {"ENOS"}, "IP_T_MIN_PU_CS"),
    ("explain LV_DT_OPEN_RLS_MT", {"TRSM"}, "LV_DT_OPEN_RLS_MT"),
    ("what is C_FLOW_WG_SP_MAN", {"CHRG"}, "C_FLOW_WG_SP_MAN"),
    ("what does NC_FID_PROD_TRPT_CH_OFF_EXTC control", {"EXTC"}, "NC_FID_PROD_TRPT_CH_OFF_EXTC"),
    # --- ordinary tuning questions -----------------------------------------
    ("how does knock control work", {"KNCK"}, ""),
    ("which tables for egt limits", {"EGCP", "EGTR", "EXTC"}, ""),
    ("what maps set lpfp pressure", {"FULP", "FUSL"}, ""),
    ("how is lambda adaptation done", {"LACO", "LASP"}, ""),
    ("what is the oil pressure switching logic", {"ENLU"}, ""),
    ("how is volumetric efficiency calculated", {"INSY"}, ""),
    ("what limits torque on simos 18.10", {"TQSP", "TQLO", "TQDR", "PTQM"}, ""),
    ("where is the ignition timing table", {"IGSP", "IGRE", "KNCK"}, ""),
    ("what controls idle speed", {"ENSC"}, ""),
    ("what maps for cam timing", {"VVTI", "VVLI"}, ""),
    ("which table sets injector deadtime", {"INJR", "FMSP"}, ""),
    ("how does the misfire detection work", {"MISF"}, ""),
    ("what controls the evap purge valve", {"EVAC", "EVAM"}, ""),
    ("how does cold start enrichment work", {"ENSS", "ENSD", "FMSP", "INJR"}, ""),
    ("what maps control the throttle position", {"THRO", "TQSP"}, ""),
    ("how is boost pressure controlled", {"CHRG", "CHRA"}, ""),
    ("what is the overrun fuel cutoff logic", {"ENOS", "TQDR", "FMSP"}, ""),
    ("which parameters set the rev limiter", {"ENOS", "TQSP", "ENSL", "PTQM"}, ""),
    ("how does the coolant thermostat control work", {"ENTE", "ECOP"}, ""),
    ("what handles the fuel tank level signal", {"FUTL"}, ""),
    ("how is the catalyst efficiency diagnosed", {"EGTR", "EXTC", "OBDC"}, ""),
    ("what controls the high pressure fuel pump", {"FUSL", "FULP"}, ""),
    ("how does the immobiliser work", {"IMMO"}, ""),
    ("what is the vehicle speed limiter", {"VHSL", "VHSC", "VHSD"}, ""),
    ("how does start stop decide to shut off", {"ENSS", "STSY"}, ""),
]

# These must never both fire and clear the gate.
NEGATIVES = [
    "whats for lunch",
    "my cat is asleep on the keyboard",
    "what time is the movie",
    "that shift at work was long",
    "who won the game last night",
    "is pizza good",
    "good morning everyone",
    "lol that bot is dumb",
]


async def run(verbose: bool) -> int:
    settings = Settings.load()
    index = frsearch.FRIndex(settings.fr_index_dir, settings.ollama_host)
    print(f"index: {len(index.chunks)} chunks, {len(index.params)} parameters, "
          f"{len(index.symbols)} symbols, gate {settings.fr_min_score}\n")

    fired = hit_label = hit_exact = exact_total = 0
    misses: list[str] = []
    for question, want_labels, want_param in CASES:
        triggers = await index.should_lookup(question)
        block, best = await index.build_context(
            question, top_k=settings.fr_top_k, min_score=settings.fr_min_score
        )
        fired += bool(triggers and block)
        labels_found = {lab for lab in want_labels if lab in block}
        ok_label = bool(labels_found)
        hit_label += ok_label
        ok_param = True
        if want_param:
            exact_total += 1
            ok_param = want_param in block
            hit_exact += ok_param
        flag = "OK " if (triggers and block and ok_label and ok_param) else "MISS"
        if flag == "MISS":
            misses.append(question)
        print(f"  {flag} fire={'Y' if triggers else 'n'} {best:.3f} "
              f"{sorted(labels_found) or '-'}"
              f"{' +' + want_param if want_param and ok_param else ''}"
              f"  | {question[:58]}")
        if verbose and block:
            print("        " + block[:300].replace("\n", "\n        "))

    print(f"\ntriggered      : {fired}/{len(CASES)}")
    print(f"right function : {hit_label}/{len(CASES)}")
    if exact_total:
        print(f"exact parameter: {hit_exact}/{exact_total}")

    print("\nnegatives (must stay quiet):")
    bad = 0
    for question in NEGATIVES:
        triggers = await index.should_lookup(question)
        block, best = await index.build_context(
            question, top_k=settings.fr_top_k, min_score=settings.fr_min_score
        )
        noisy = bool(triggers and block)
        bad += noisy
        print(f"  {'**FIRED**' if noisy else 'quiet'} {best:.3f}  {question}")
    print(f"\nfalse positives: {bad}/{len(NEGATIVES)}")
    if misses:
        print("\nstill missing:")
        for m in misses:
            print(f"  - {m[:76]}")
    return 0 if not bad else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    sys.exit(asyncio.run(run(args.verbose)))
