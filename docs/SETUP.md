# Setup

This walks through getting Preston running from nothing. The basic bot (chat,
personas, commands, the economy) needs only Python, a Discord token and a model.
Pictures, songs and retrieval are optional extras layered on top.

- [1. Create the Discord application](#1-create-the-discord-application)
- [2. Install](#2-install)
- [3. Choose a model backend](#3-choose-a-model-backend)
- [4. Configure `.env`](#4-configure-env)
- [5. Run it](#5-run-it)
- [6. Optional: pictures and songs (GPU)](#6-optional-pictures-and-songs-gpu)
- [7. Optional: retrieval indexes](#7-optional-retrieval-indexes)
- [8. Optional: web search](#8-optional-web-search)
- [Troubleshooting](#troubleshooting)

---

## 1. Create the Discord application

1. Open the [Discord Developer Portal](https://discord.com/developers/applications) and click **New Application**.
2. **Bot** tab:
   - **Reset Token** and copy it. This is `DISCORD_TOKEN`; treat it like a password.
   - Under **Privileged Gateway Intents**, enable **MESSAGE CONTENT INTENT**. (The
     members intent is not needed; names are resolved as people talk.)
3. **OAuth2 → URL Generator**:
   - Scopes: `bot`, `applications.commands`
   - Permissions: View Channels, Send Messages, Send Messages in Threads, Embed Links,
     Attach Files, Read Message History, Add Reactions, Use Slash Commands
   - Open the generated URL and add the bot to your server.
4. In Discord, turn on **Developer Mode** (User Settings → Advanced), right-click your
   server and **Copy Server ID**. That is `DISCORD_GUILD_ID`; with it set, slash
   commands appear instantly instead of after up to an hour.

## 2. Install

Python 3.12 or newer (developed on 3.14).

```powershell
git clone https://github.com/subtle43/preston-sterling-bot.git
cd preston-sterling-bot
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env
```

On Linux or macOS use `python3 -m venv .venv`, `.venv/bin/python` and `cp`.
`run.bat` does the venv and install for you on Windows the first time.

## 3. Choose a model backend

Preston talks to one of two backends through the same interface
(`gemini_client.py` and `ollama_client.py`), so everything else works the same.

**Gemini (cloud, recommended for the full experience).** Big context window, fast,
and it powers the intent classifier and structured JSON for the gag commands.

```env
GEMINI_API_KEY=your_key_from_aistudio.google.com
GEMINI_MODEL=gemini-3.5-flash-lite
```

**Ollama (fully local).** Nothing leaves your machine. Smaller models do simpler,
shorter replies; the bot automatically uses tighter budgets for them.

```powershell
ollama pull gemma4:e4b
```

```env
GEMINI_MODEL=
OLLAMA_MODEL=gemma4:e4b
```

You can switch between them live with `/model` (owner only). That choice is
saved in `data/runtime.json` and survives restarts.

## 4. Configure `.env`

Every setting is documented inline in [`.env.example`](../.env.example). The
important ones:

| Setting | What it does |
| --- | --- |
| `DISCORD_TOKEN` | Required. |
| `DISCORD_GUILD_ID` | Your server, for instant slash-command sync. |
| `OWNER_IDS` | Your Discord user ID. Unlocks owner commands (`/model`, `/settle`, `!clear`, `!why`...). |
| `ALLOWED_GUILD_IDS` | Restrict the bot to your server(s). |
| `REPLY_CHANNEL` | If set, every reply lands in this one channel, wherever he was pinged. |
| `HARNESS` | `full` (whole persona prompt, the default), `lite` (short prompt, for small local models), `voice` (for a fine-tuned model). |
| `LORE_ENABLED`, `PERSIST_MEMORY` | Remember people and conversations across restarts (in `data/`). |
| `DAILY_RAP_CHANNEL`, `WEEKLY_AWARDS_*` | Scheduled posts. |

## 5. Run it

```powershell
.\.venv\Scripts\python bot.py
```

or double-click `run.bat`. When the log shows `Logged in as ...` and
`Synced slash commands`, @mention the bot or try `/ask`.

Everything the bot learns while running (memory, lore, balances, the active
persona, moods) is written under `data/`, which is created on first run and never
committed.

Choose a personality with `/persona` (leave the name empty to see the list with
descriptions).

## 6. Optional: pictures and songs (GPU)

Both run locally through Hugging Face `diffusers` on an NVIDIA GPU. 8 GB of VRAM
is enough; the two models never sit in memory together. Each loads on first use
and unloads after a couple of idle minutes, so the card is free the rest of the time.

Install PyTorch for your CUDA version first (see [pytorch.org](https://pytorch.org)), then:

```powershell
.\.venv\Scripts\python -m pip install "diffusers>=0.40" transformers accelerate gguf
```

| Setting | Default | Notes |
| --- | --- | --- |
| `IMAGE_GEN_ENABLED` | `true` | Master switch for pictures and photo edits. |
| `IMAGE_LOCAL_MODEL` | `Tongyi-MAI/Z-Image-Turbo` | Downloaded from Hugging Face on first use (several GB). |
| `SONG_GEN_ENABLED` | `true` | Master switch for sung songs. |
| `SONG_MODEL` | `ACE-Step/acestep-v15-xl-turbo-diffusers` | ACE-Step 1.5; several GB on first use. |
| `SONG_DURATION` | `60` | 10-180 seconds. The lyric structure scales with it. |
| `DAILY_RAP_AUDIO` | `false` | Also sing the morning rap. |

Smoke tests without Discord:

```powershell
.\.venv\Scripts\python localimage.py "a mk7 gti on a rusty trailer"
.\.venv\Scripts\python songgen.py "country rap, male vocals, banjo, 110 bpm" -o data\test.mp3
```

> Tip: switch both off (`IMAGE_GEN_ENABLED=false`, `SONG_GEN_ENABLED=false`) while
> you rebuild an embedding index on the same GPU.

## 7. Optional: retrieval indexes

Preston can search your server's chat history and your ECU documentation. None of
that data is in this repository; you build it yourself. It needs an embedding model
in Ollama even if you chat through Gemini:

```powershell
ollama pull bge-m3
```

Then follow [RAG.md](RAG.md).

## 8. Optional: web search

Get a free key from [Tavily](https://tavily.com) (1,000 searches a month), then:

```env
TAVILY_API_KEY=tvly-...
SEARCH_ENABLED=true
```

Results are treated as untrusted data: links are stripped, and the model is told
never to follow instructions found in them.

---

## Troubleshooting

**`Missing DISCORD_TOKEN`**: create `.env` from `.env.example` and paste the token.

**`Discord login failed`**: the token is wrong or was reset. Copy it again.

**`PrivilegedIntentsRequired`**: enable MESSAGE CONTENT intent in the Developer
Portal → Bot, then restart. Or set `MESSAGE_CONTENT_INTENT=false`
and use only slash commands and @mentions.

**Slash commands missing**: set `DISCORD_GUILD_ID` and restart. Global sync can
take an hour; guild sync is immediate.

**`Ollama model '...' is not installed`**: `ollama list`, then `ollama pull <name>`.

**Online but never replies**: check the invite had the `bot` scope, the bot can
Send Messages in that channel, you @mentioned or replied to it, and the process is
still running. With `REPLY_CHANNEL` set, replies go to that channel.

**`!why`** (owner) prints the decision trail for the last reply in a channel: what
the classifier said, which gate took the message, retrieval verdicts and the reply
shape. It's the fastest way to see why the bot did something.
