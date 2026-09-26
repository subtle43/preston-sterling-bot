# Retrieval (RAG): building your own indexes

Preston can look things up in three places. **None of them ship with this
repository.** The chat archive is your community's conversation and the documents
are whatever you supply. You build each index yourself,
it lives in `data/` (gitignored), and it never leaves your machine.

| Index | Source | Built by | Searched by | Switch |
| --- | --- | --- | --- | --- |
| Server chat archive | Your Discord server's history | `scrape_chat.py` + `build_chat_index.py` | `chatsearch.py` | `CHAT_INDEX_ENABLED` |
| Funktionsrahmen (FR) | An ECU function-documentation PDF | `frextract.py` + `build_fr_index.py` + `build_fr_tables.py` | `frsearch.py` | `FR_ENABLED` |
| A2L | One or more ECU `.a2l` description files | `build_a2l_index.py` | `frsearch.py` | (loaded with the FR index) |

All three embed text with **`bge-m3`** (1024 dimensions) through Ollama, even if the
bot chats through Gemini:

```powershell
ollama pull bge-m3
```

Embedding runs on your GPU. Turn off pictures and songs (`IMAGE_GEN_ENABLED=false`,
`SONG_GEN_ENABLED=false`) while you build, so they don't fight for VRAM.

---

## 1. The server chat archive

This is what lets Preston answer "what did X say about his turbo", "when did Y blow
his rods", build member profiles for roasts, songs and trading cards, and know
everyone's car.

> **Consent first.** This stores your server's messages on your disk and feeds
> excerpts to a model (to Google, if you use Gemini). Tell your community before you
> turn it on.

### Step 1: scrape (`scrape_chat.py`)

```powershell
.\.venv\Scripts\python scrape_chat.py              # start, or resume, the crawl
.\.venv\Scripts\python scrape_chat.py --status     # what has been collected
.\.venv\Scripts\python scrape_chat.py --restart    # ignore progress, pull everything again
```

- Logs in with the same `DISCORD_TOKEN` and reads every text channel the bot can
  see (it needs **Read Message History** there). It is read-only: it never posts.
- Output: `data/chat_index/raw.jsonl` (one message per line) and `state.json` (a
  cursor per channel).
- **Resumable and incremental.** Progress is saved every 500 messages, so closing
  the window costs at most one batch. Re-running later fetches only messages newer
  than the last run, which makes refreshing the archive a matter of minutes.
- Discord rate-limits history to roughly 100 messages a second, so a first crawl of
  a large server takes hours. `crawl_all.bat` runs it on Windows.
- Messages under 15 characters ("lol", "same") and link-only messages are dropped
  at this stage. They add nothing to search and dilute the vectors.

### Step 2: build the index (`build_chat_index.py`)

```powershell
.\.venv\Scripts\python build_chat_index.py            # chunk + embed
.\.venv\Scripts\python build_chat_index.py --dry-run  # chunk and report only
```

Chat is not documentation. A single message averages about 60 characters, and a
vector for "what psi" means nothing. So messages are grouped into **conversations**:

- consecutive messages in one channel, split wherever there is a silence longer
  than **15 minutes**;
- packed to about **250 words** per chunk, with the **speaker's name inline** on
  every line so a question naming a person has something to match;
- **2 messages of overlap** between chunks, so a point made across a boundary is
  still findable;
- fragments under 25 words are dropped.

Output in `data/chat_index/`: `chunks.jsonl`, `vectors.npy` (float32, normalised),
`speakers.json` (every speaker, their message counts and **aliases**, so a member
who renames themselves keeps their history) and `meta.json`.

Embedding runs at about 40-60 chunks a second on an 8 GB card (roughly half an hour
for 80k chunks). It is safe to re-run at any time: it only reads `raw.jsonl`. The
running bot holds the old index in memory, so restart it after a rebuild.

### Step 3: turn it on

```env
CHAT_INDEX_ENABLED=true
CHAT_INDEX_ROUTER=true   # a small model call decides if a message needs the archive
```

### How search works (`chatsearch.py`)

- A **router** decides whether a message needs the archive at all. Most chat doesn't.
- Queries mix **vector similarity** with **name matching**: "what did matto say
  about X" resolves a partial or old handle to a speaker, then searches their lines.
- **Speaker tools** used across the bot: `speaker_lines` (what someone said),
  `speaker_topic` (what someone said about a subject), `build_profile` (a summary of
  a member: their car, their takes, their running arguments) and `image_sketch` (a
  visual description for portraits).
- Everything retrieved is fenced as **untrusted** before it reaches the model, and
  the privacy rules in the shared persona rules apply: cars, builds and takes are
  fair game; where people live, work, family and health are not.

### Keeping it fresh

```powershell
.\.venv\Scripts\python scrape_chat.py        # minutes: only new messages
.\.venv\Scripts\python build_chat_index.py   # re-embeds everything
# then restart the bot
```

---

## 2. The Funktionsrahmen (FR)

A Funktionsrahmen is the ECU supplier's function documentation: thousands of pages
describing every function, label and calibration parameter. Preston can quote it
by label (`LACO`, `IP_T_MIN_PU_CS`) or by plain-English question.

This repository contains **only the tools**, not the document or any index built
from it.

### Build

```powershell
# 1. Extract page text once (the slow, fragile step)
.\.venv\Scripts\python frextract.py "path\to\your FR.pdf" --out data\fr_pages.jsonl

# 2. Prose index: chunks.jsonl, vectors.npy, vocab.json, meta.json
.\.venv\Scripts\python build_fr_index.py

# 3. Structured tables: symbols.json (label -> defining/using functions)
#    and params.jsonl/params.npy (one record per calibration parameter)
.\.venv\Scripts\python build_fr_tables.py
```

Why three steps: extraction reads a 200+ MB PDF and is slow, so it's done once.
Chunking and embedding can then be re-run in minutes. And the FR isn't really prose.
It's a generated reference (a symbol table plus data-definition tables laid out as
PDF). `frparse.py` reconstructs that structure, so a parameter's one-line description
is found on its own instead of drowning in a page of hex ranges.

### Turn it on and tune it

```env
FR_ENABLED=true
FR_INDEX_DIR=data/fr_index
FR_TOP_K=8
FR_MIN_SCORE=0.50   # raise it if the bot quotes the FR at irrelevant moments
```

A gate decides whether a question is FR-shaped at all; exact label matches always
win over vectors. `/fr <query>` searches it directly.

### Measure before you change retrieval

```powershell
.\.venv\Scripts\python eval_fr.py            # recall and false positives on a fixed question set
.\.venv\Scripts\python eval_fr.py --verbose
.\.venv\Scripts\python build_fr_review.py    # sample real questions from your chat archive for human review
```

`eval_fr.py` is a regression suite: questions paired with the function (and
sometimes the exact parameter) that should come back. Edit its cases for your own
document. `build_fr_review.py` pulls real tuning questions from your chat archive
and records what the FR search returns for each, for a person to mark right or
wrong. The marks become the next set of eval cases.

---

## 3. A2L files

An A2L describes an ECU's calibration characteristics, measurements and axes. With
it indexed, Preston can name the actual map behind a question.

```powershell
.\.venv\Scripts\python build_a2l_index.py "path\to\first.a2l=ECU_A" "path\to\second.a2l=ECU_B"
```

The `=TAG` after each path names its ECU. Every record keeps that tag and it's
shown on every line the model sees, because a calibration from the wrong ECU is
worse than no answer. Output: `data/fr_index/a2l.jsonl`, `a2l.npy`, `a2l_meta.json`.

`a2lparse.py` is a standalone ASAM A2L parser if you want the records for something else.

---

## Troubleshooting

- **`model "bge-m3" not found`**: `ollama pull bge-m3`.
- **Out of GPU memory while embedding**: switch pictures and songs off, or stop other GPU apps.
- **The bot doesn't see a rebuilt index**: indexes load at start-up, so restart the bot.
- **Search finds nothing for a person**: check `speakers.json` for how their name
  was indexed. Renames are merged by user ID at build time.
