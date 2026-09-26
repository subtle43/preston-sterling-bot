# Preston Sterling: a personality-driven AI bot for any Discord server

Preston is an AI community bot for Discord. He answers questions properly, learns
your server's history and its people, draws pictures, writes and *sings* songs
about your members, runs a fake-money betting economy and a stack of gag
commands, and does all of it in whichever of 40+ swappable personalities the
server has picked: a film-noir detective, a pirate, a Bond villain, a sweet grandma,
a divorced dad who overshares, a drill sergeant...

The joke is always the delivery. The answer underneath is meant to be real.

**It works for any community**: gaming, a hobby, a friend group, a fandom, a
study server. The personalities, rules and expertise are plain text files you
edit, not code. It was first built for a car-tuning server, so the example
personas talk about cars and it ships optional tuning modules (datalog review,
ECU documentation lookup). Switch those off, point the personas at your own topic,
and it's your server's bot. See [Making it yours](#making-it-yours).

> This repository is the code only. It ships **no** API keys, **no** chat
> messages, **no** retrieval indexes and **no** third-party documents. You bring
> your own tokens, and you build your own indexes from your own server and your
> own documents. [docs/RAG.md](docs/RAG.md) explains how.

---

## What it does

| Area | What Preston does |
| --- | --- |
| **Chat** | Replies when @mentioned, replied to, DMed, or via `/ask`. Keeps per-channel memory that folds into a running summary, remembers facts about people (`!remember`), and has moods that change his delivery for an hour or two. |
| **Personas** | 40+ personalities in plain text files under [`prompts/personas/`](prompts/personas/README.md). Anyone can switch with `/persona` (5-minute cooldown). Each persona can also set a music genre for its songs and a look for its pictures. Write your own in a few minutes. |
| **Server memory (RAG)** | Searches the server's whole chat history to answer "what did X say about Y" or "when did Z happen", and builds member profiles for roasts, songs and cards. |
| **Pictures** | "draw ...", "make a meme of ...", or an edit under a photo ("make it look terrible"). Generated locally on your GPU (Z-Image Turbo through diffusers). |
| **Songs** | "make a song about @member" writes lyrics from the server's history and sings them with a local ACE-Step model, with a generated album cover. |
| **Gags** | `/sue` (a mock trial), `/tierlist` (rank the regulars on anything), `/card` (member trading cards), `/stock`, `/factcheck`, `/race`, `/dyno`, `/wordle`, weekly awards and a daily rap roasting a random member. |
| **Server economy** | A fake-money economy with the bot as a crooked bookie: `/daily`, `/bet`, `/odds`, markets it opens on its own from things people claim in chat, `/leaderboard`. |
| **Document lookup (RAG)** | Index a reference document and the bot quotes it by name or plain-English question. Built for a technical manual; the extractor can be adapted to others. |
| **Web search** | Optional, via Tavily. Results are treated as untrusted and links are stripped. |
| **Optional: car tuning** | The modules it was born with: datalog review (attach a CSV: finds the pulls, checks boost, timing, knock and fuel, posts a chart), ECU documentation and A2L lookup, `!car`. Harmless if unused. |

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

## Making it yours

Everything that makes Preston a *car* bot is text, not code. To make him the bot
for your gaming clan, book club or friend group:

1. **Give him your topic.** Open [`prompts/personas/_shared/rules.txt`](prompts/personas/_shared/rules.txt)
   and rewrite the "WHAT YOU ACTUALLY KNOW" section with your community's subject:
   Valorant, sourdough, Warhammer, anything. Each persona's opening lines also say
   what he does for a living ("a SIMOS18 tuner"); a find-and-replace across
   `prompts/personas/` changes it everywhere.
2. **Pick or write personalities.** Copy any persona folder and rewrite three short
   text files. See the [persona guide](prompts/personas/README.md). Switch live with
   `/persona`; no restart, no code.
3. **Teach him your server.** Scrape and index your own chat history
   ([docs/RAG.md](docs/RAG.md)) and he learns who's who, the running jokes and the
   old arguments. That's what makes the roasts, songs, trading cards and trials
   personal.
4. **Add your reference material (optional).** The document-lookup pipeline (chunk,
   embed with `bge-m3`, search with a relevance gate) is generic. Its PDF extractor
   (`frextract.py`) is written for one ECU manual's page layout, so for a rulebook,
   a wiki export or a manual you'd adapt that one file to your document.
5. **Leave the car bits off.** Document lookup is off by default, and log review
   only runs when someone attaches a datalog. A few gags are car-flavoured (`/dyno`,
   `/race`, the horsepower `/stock`); they still work as jokes, or you can delete
   them from `bot.py`.

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
frsearch.py  frparse.py  a2lparse.py   retrieval over reference documents (built for ECU docs)
logtrack.py  logpulls.py  logchart.py  datalog review and charts (car tuning, optional)
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
