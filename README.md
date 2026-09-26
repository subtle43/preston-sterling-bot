# Preston Sterling: a Discord bot for a car-tuning server

Preston is a Discord bot built for a Volkswagen/Audi ECU tuning community
(SIMOS18, EA888, IS38 and friends). He answers tuning questions properly, reads
datalogs, looks things up in the ECU documentation and the server's own chat
history, draws pictures, writes and *sings* songs, runs a fake-money betting
economy, and does all of it in whichever of 40+ swappable personalities the
server has picked: a divorced shop dad, a film-noir detective, a pirate, a Bond
villain, an 84-year-old ex-Bosch engineer grandma...

The joke is always the delivery. The answer underneath is meant to be real.

> This repository is the code only. It ships **no** API keys, **no** chat
> messages, **no** retrieval indexes and **no** ECU documentation. You bring your
> own tokens, and you build your own indexes from your own server and your own
> documents. [docs/RAG.md](docs/RAG.md) explains how.

---

## What it does

| Area | What Preston does |
| --- | --- |
| **Chat** | Replies when @mentioned, replied to, DMed, or via `/ask`. Keeps per-channel memory that folds into a running summary, remembers facts about people (`!remember`, `!car`), and has moods that change his delivery for an hour or two. |
| **Personas** | 40+ personalities in plain text files under [`prompts/personas/`](prompts/personas/README.md). Anyone can switch with `/persona` (5-minute cooldown). Each persona can also set a music genre for its songs and a look for its pictures. |
| **Tuning help** | Datalog review (attach a CSV): finds the pulls, checks boost, timing, knock, lambda and fuel against fixed limits, and posts a chart. Looks things up in an ECU Funktionsrahmen and A2L files if you index your own. |
| **Server memory (RAG)** | Searches the server's whole chat history to answer "what did X say about Y" or "when did Z blow his turbo", and builds member profiles for roasts, songs and cards. |
| **Pictures** | "draw ...", "make a meme of ...", "shittify this car" under a photo. Generated locally on your GPU (Z-Image Turbo through diffusers). |
| **Songs** | "make a song about ..." writes lyrics from the server's history and sings them with a local ACE-Step model, with a generated album cover. |
| **Gags** | `/dyno`, `/sue` (Tuning Court), `/race`, `/tierlist`, `/stock`, `/factcheck`, `/card`, `/wordle`, weekly awards and a daily rap about a random member. |
| **Preston Bucks** | A fake-money economy: `/daily`, `/bet`, `/odds`, markets that the bookie opens on its own from chat, `/leaderboard`. |
| **Web search** | Optional, via Tavily. Results are treated as untrusted and links are stripped. |

The full list is in [docs/COMMANDS.md](docs/COMMANDS.md).

## How it fits together

```
Discord message
   │
   ├─ commands (/slash, !prefix)  ─────────────► handlers in bot.py
   │
   └─ addressed to Preston?
         │
         ├─ intent classifier (one small Gemini call, keyword gates as fallback)
         │     picture? edit? song? log? question? needs the web/archive/FR?
         │
         ├─ media paths ── imagegen.py / localimage.py (pictures)
         │                 songgen.py (songs), logtrack.py + logpulls.py (logs)
         │
         └─ chat reply
               system prompt  = active persona (persona.py) + shared rules
               + member lore, channel memory/summary   (lore.py, memory.py)
               + retrieved material                     (chatsearch.py, frsearch.py, websearch.py)
               + mood and reply shape                   (mood.py, feedback.py)
               + per-reply voice note (what the model reads last)
               │
               └─ model backend: Gemini API or local Ollama
                  (gemini_client.py and ollama_client.py share one interface)
```

The design is covered in detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start

You need **Python 3.12+**, a **Discord bot token**, and **either** a Gemini API
key **or** [Ollama](https://ollama.com) running locally.

```powershell
git clone https://github.com/subtle43/preston-sterling-bot.git
cd preston-sterling-bot
copy .env.example .env          # then fill in DISCORD_TOKEN and DISCORD_GUILD_ID
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python bot.py   # or double-click run.bat on Windows
```

For Gemini, set `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env`. For Ollama, run
`ollama pull gemma4:e4b` and leave `GEMINI_MODEL` blank.

Pictures and songs need a CUDA GPU (8 GB works) and a few extra packages.
Retrieval needs `ollama pull bge-m3` for embeddings. Both are covered step by
step in [docs/SETUP.md](docs/SETUP.md).

## Documentation

| Doc | What's in it |
| --- | --- |
| [docs/SETUP.md](docs/SETUP.md) | Discord app, install, Gemini vs Ollama, GPU extras, running it, troubleshooting |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How a message becomes a reply: routing, harnesses, memory, moods, personas, media, scheduled jobs |
| [docs/RAG.md](docs/RAG.md) | Building your own retrieval indexes: server chat archive, Funktionsrahmen, A2L |
| [docs/COMMANDS.md](docs/COMMANDS.md) | Every slash command, prefix command and natural-language trigger |
| [prompts/personas/README.md](prompts/personas/README.md) | How personas work and how to write a new one |
| [.env.example](.env.example) | Every setting, with comments |

## Project layout

```
bot.py                  the Discord client: routing, replies, commands, scheduled jobs
config.py               every setting, read from .env
persona.py              the active personality and its files
gemini_client.py        Gemini backend        ollama_client.py   Ollama backend
intent.py               what a message wants (picture, song, question...)
memory.py  lore.py      channel memory + summaries; facts about members
mood.py  feedback.py    moods; which reply shapes the room reacts to
chatsearch.py           retrieval over the server's chat history
frsearch.py  frparse.py  a2lparse.py   retrieval over ECU documentation
logtrack.py  logpulls.py  logchart.py  datalog review and charts
imagegen.py  localimage.py  songgen.py  pictures and songs
bucks.py  gags.py  dynochart.py  wordleplay.py   the economy and the gag commands
websearch.py            Tavily web search
scrape_chat.py  build_chat_index.py              build the chat archive index
frextract.py  build_fr_tables.py  build_fr_index.py  build_a2l_index.py  eval_fr.py
prompts/personas/       one folder per personality
tests/                  intent and routing tests
```

## What is deliberately not here

- **Secrets.** `.env` is gitignored. Only `.env.example`, with blank values, is committed.
- **Chat data.** No messages, member lists, memory, lore or balances. Everything the
  bot learns lives in `data/`, which is gitignored.
- **Indexes and documents.** The Funktionsrahmen and A2L files are proprietary
  manufacturer documents and are not included. Neither is any built index.
- **The fine-tuning pipeline** used to train a model on one member's voice
  (with their consent) is not included.

## A note on the humour

Personas roast people, swear and get weird, but every persona file carries the
same hard lines: no slurs, nothing about race, religion or sexuality, and no
jokes about sexual abuse. The shared rules in
[`prompts/personas/_shared/rules.txt`](prompts/personas/_shared/rules.txt)
also require every reply to contain the real answer, and forbid inventing ECU
labels, numbers or quotes.

## License

MIT. See [LICENSE](LICENSE).
