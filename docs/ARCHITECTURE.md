# Architecture

How a Discord message turns into a reply, a picture, a song or a log review, and
where each piece lives. The core (routing, personas, memory, retrieval, media, gags,
the economy) is topic-neutral. The car-tuning parts (datalog review, ECU document
and A2L lookup) are separate modules that only run when used. The code is heavily commented with the *why* behind each
decision (usually the bug that caused it), so this page is a map and the source is
the territory.

- [The big picture](#the-big-picture)
- [Model backends](#model-backends)
- [Routing: what does this message want?](#routing-what-does-this-message-want)
- [Building a chat reply](#building-a-chat-reply)
- [Harnesses: full, lite, voice](#harnesses-full-lite-voice)
- [Personas](#personas)
- [Memory, lore and summaries](#memory-lore-and-summaries)
- [Moods, reply shapes and feedback](#moods-reply-shapes-and-feedback)
- [Retrieval](#retrieval)
- [Datalog review](#datalog-review)
- [Pictures](#pictures)
- [Songs](#songs)
- [Gag commands and structured output](#gag-commands-and-structured-output)
- [Preston Bucks](#preston-bucks)
- [Scheduled jobs](#scheduled-jobs)
- [Safety and privacy](#safety-and-privacy)
- [Where state is kept](#where-state-is-kept)

---

## The big picture

`bot.py` holds one `commands.Bot` subclass (`OllamaBot`) plus the slash commands.
Everything else is a module it calls into. Almost nothing is hard-coded to one
model or one personality: the backend is swappable, the persona is a folder of
text files, and every prompt that used to name a tone now says "in your persona's
voice".

```
on_message ─► commands? ─► handler
            └► addressed to the bot? ─► intent ─► picture / edit / song / log / chat
                                                             │
                           persona + rules + lore + memory + retrieval + mood + voice note
                                                             │
                                                  Gemini or Ollama (streamed)
                                                             │
                                  clean-up (strip leaked stage directions, length) ─► Discord
```

## Model backends

`gemini_client.py` (`GeminiChat`) and `ollama_client.py` (`OllamaChat`) expose the
same methods (`build_messages`, `stream_chat`), so `bot.py` never knows which one
it's talking to. Setting `GEMINI_MODEL` selects Gemini; blank means Ollama. The
owner can flip it live with `/model`, which also updates the bot's presence line.

Details worth knowing:

- **Budgets follow the backend.** Gemini gets hundreds of messages of history and a
  large context block; a small local model gets a few and a shorter word cap
  (`REPLY_MAX_WORDS_LOCAL`). See the `*_GEMINI` settings.
- **Retries.** Transient Gemini errors (429/5xx, "model overloaded") are retried,
  and a request can fall back to `GEMINI_FALLBACK_MODEL` when the chosen model is
  saturated.
- **Structured output.** `llm_json(system, user, schema)` asks for one JSON object.
  On Gemini the API enforces the schema (`response_schema`); on a local model it
  asks nicely and parses leniently. All the gag commands use this.
- **Heavy model.** With `OLLAMA_MODEL_HEAVY` set, log reviews and long answers go to
  a bigger local model.

## Routing: what does this message want?

The bot replies when it's @mentioned, replied to, DMed, used through `/ask`, or in a
`LISTEN_CHANNELS` channel. Ambient behaviour (the odd reaction or interjection,
auto-roasts of `ROAST_USER_IDS`) is rate-limited and can be confined to
`AMBIENT_CHANNELS`.

For a message addressed to it, `intent.py` makes **one small classifier call**
(Gemini, minimal thinking) that reads the message the way a person would: is this a
picture, an edit of a picture in play, a meme, a song/rap/poem, a summary request, a
question, and does it need the web, the server archive or the ECU docs? It also
pulls out a style ("as a sea shanty") and spots a second request in the same
sentence ("draw his car and write a song about it").

Keyword gates stay in place as the fast path and as the fallback when the
classifier is off (`INTENT_CLASSIFIER=false`) or a local model is in use. The
classifier can open a gate for phrasing the regexes miss, and close one: "paint the
calipers red or black?" is a question, not a commission.

Follow-ups work within a 30-minute window: "make it darker", "again but as a
cartoon" and "same song but country" act on the last thing made in that channel.

`!why` (owner) prints the whole decision trail for the last reply in a channel.

## Building a chat reply

The system prompt is assembled per reply, roughly in this order:

1. **The persona** (`persona.read("system")`): the character, followed by the shared
   rules in `prompts/personas/_shared/rules.txt`: answer first, never refuse, match
   length to the question, never invent ECU labels or numbers, privacy, what the bot
   can do.
2. **Who's talking**: facts from `lore.py` (their car, notes they've asked it to
   remember) and, for members in the archive, a profile built from their history.
3. **The channel**: recent messages verbatim plus a rolling summary of older ones.
4. **Material**: retrieved chat history, Funktionsrahmen/A2L hits, web results or the
   last log review. Every block is labelled as untrusted where it comes from users
   or the web.
5. **Delivery**: the channel's current mood, a reply "shape" and variety hints.
6. **The voice note**: one paragraph from the persona's `voice_note.txt`, placed
   right before `[NOW REPLY TO THIS MESSAGE...]` and the question itself.

The order matters. Models obey what they read last, so the voice note and the
question sit at the very end. That fixed a long list of "drifted out of character"
and "answered the wrong question" bugs.

The reply streams back and is cleaned: leaked stage directions and "thinking" are
stripped, overlong replies are clamped, and a reply that only repeats the question
is caught.

## Harnesses: full, lite, voice

`HARNESS` decides how much prompt wraps an ordinary chat reply:

| Harness | Prompt | For |
| --- | --- | --- |
| `full` | Everything above | Gemini and capable local models (default) |
| `lite` | The persona's short `lite.txt`, the last few messages, at most one fact block | Small local models that drown in a long prompt |
| `voice` | One line plus the last ~8 messages | A model fine-tuned on chat, which has the voice baked in |

Pictures, songs and log reviews always use the full path. `/model` can switch
harness live.

## Personas

`persona.py` keeps the active persona name in `data/persona.json` and reads its
files fresh every time, so editing a file or running `/persona` applies on the very
next reply with no restart.

A persona is a folder in `prompts/personas/<name>/`:

| File | Used for |
| --- | --- |
| `system.txt` | The character (the full harness). The shared rules are appended automatically unless the file carries its own copy. |
| `lite.txt` | Short version: the lite harness and the voice of every gag command. |
| `voice_note.txt` | The last thing the model reads on each reply. |
| `about.txt` | One line for the `/persona` list and search. |
| `song_style.txt` | Optional music genres for its songs. |
| `image_style.txt` | Optional default look for its pictures. |

`/persona` is open to everyone with a 5-minute cooldown (the owner is exempt), and
each switch is announced in the channel. `/persona random` picks one. The morning
rap is written by a random persona each day without changing the live one.

Lessons baked into how personas are written (see the
[persona README](../prompts/personas/README.md)): example lines get copied word for
word, so they're written about unrelated topics; the voice note must ask for
rotation or every reply opens the same way; and every persona keeps the same hard lines.

## Memory, lore and summaries

- **`memory.py`: channel memory.** Per channel (or DM), recent messages verbatim
  plus a rolling summary. When the verbatim window overflows, older messages are
  *folded* into the summary by a background model call, so a long argument is still
  remembered in outline. Persisted to `data/history.json` with `PERSIST_MEMORY`.
  `!memory` shows what's held; the owner commands `!clear` and `!compact` wipe or
  refold it.
- **`lore.py`: facts about people.** `!car MK7 GTI, IS38, E30` and
  `!remember I run 93 and a stage 2 tune` store short, capped, flattened notes per
  user (flattened so stored text can never forge a prompt section later). Log
  reviews also record a member's latest numbers. `!whois` shows what's stored and
  `!forget` deletes everything held about you.
- **Callbacks.** Now and then the bot brings up something a person said before.
  Each callback retires after a couple of uses so it doesn't become a catchphrase.

## Moods, reply shapes and feedback

- **`mood.py`.** Each channel has a mood that lasts 30-120 minutes (weary, manic,
  philosophical...). Moods change *delivery only*, never the persona or the facts,
  and each one ends by reminding the model to stay in character.
- **Reply shapes.** Each reply is rolled a shape (a straight answer, an anecdote, a
  one-liner...) so the bot doesn't write the same kind of reply every time.
  `STRAIGHT_CHANCE` controls how often it simply answers without a bit.
- **`feedback.py`.** Reactions on the bot's replies are scored per shape (laughing
  emoji count extra), and the shape roll leans toward what this particular room
  laughs at. `!feedback` shows the table.

## Retrieval

Three separate indexes, all embedded with `bge-m3` through Ollama and searched with
a mix of vector similarity and exact identifier matching. None ship with the repo;
[RAG.md](RAG.md) covers building them.

- **Server chat archive** (`chatsearch.py`). The whole history, chunked into
  *conversations* rather than single messages. A router call decides whether a
  message needs it at all. It also powers per-member profiles, speaker lookups
  ("what did X say about turbos"), handle renames and the gag commands.
- **Funktionsrahmen** (`frsearch.py`, `frparse.py`). The ECU function
  documentation, rebuilt from its PDF into prose chunks, a symbol table (which
  function defines or uses each label) and a parameter dictionary. A retrieval gate
  and boilerplate clean-up keep it from quoting the FR at irrelevant moments;
  `FR_MIN_SCORE` sets the bar.
- **A2L** (`a2lparse.py`). Calibration characteristics and measurements from one or
  more A2L files, each tagged with its source ECU so a map from the wrong ECU is
  never presented as the answer.

## Datalog review

Attach a `.csv` log and ask about it:

1. `logtrack.py` reads the log the way a tuner does, across each pull rather than
   as averages, and maps the many column-naming schemes to known channels.
2. `logpulls.py` finds the wide-open-throttle pulls and checks each one against
   fixed limits in code (boost vs target, timing and knock retard, lambda, fuel
   pressures, IATs). The model never has to spot a problem itself; the findings are
   handed to it as facts.
3. `logchart.py` draws the reviewed pull (matplotlib, in a worker thread) and posts it
   under the review.

The last review in each channel is kept, so "how does that compare" works as a
follow-up.

## Pictures

`imagegen.py` decides and prepares; `localimage.py` renders on the GPU with
diffusers. The persona writes the image prompt, but it's told to draw what was asked
*straight*: its opinions go in the caption. If a request names no style, the
persona's `image_style.txt` is used (noir is black-and-white film, the pirate is an
old engraving); otherwise photorealistic. A picture of a member is a portrait built
from their archive: what they drive and obsess over, never personal details. Edits
("shittify this car" under a photo) run image-to-image. The model loads on first use
and unloads when idle.

## Songs

`write_rap` / `write_topic_song` write lyrics from the archive in the persona's
voice, in a genre from the persona's `song_style.txt` (or a random one, or the one
requested). `song_brief` then asks for a title, the music-generator tags and cover
art in one call, and `songgen.py` sings it with ACE-Step 1.5. The lyric structure is
sized to `SONG_DURATION`, because the model sings roughly a line every three seconds.
Songs are posted as MP3 with the lyrics and a generated cover.

## Gag commands and structured output

`/dyno`, `/sue`, `/race`, `/tierlist`, `/stock`, `/factcheck`, `/card` and `/wordle`
follow one pattern:

1. Build a **dossier** of the member from the archive and lore (`member_dossier`).
2. Ask the model for **one JSON object** matching a schema, with the persona's
   `lite.txt` as the voice (`llm_json`).
3. Render it in code: matplotlib in a worker thread (`dynochart.py`, `gags.py`) for
   dyno sheets, time slips, tier boards, stock charts and trading cards.

So the numbers and layout are always valid, and the jokes are always in the current
persona's voice. `/wordle` (`wordleplay.py`) plays the day's puzzle honestly: the code
fetches the answer and scores each guess exactly as the game does, but the model
only ever sees the colours. It posts a spoiler-free grid.

## Preston Bucks

`bucks.py` is a JSON-backed ledger (`data/bucks.json`) with the bot as a crooked
bookmaker: `/daily` claims, fractional odds, a 5% house cut, and persistent bet
buttons (`discord.ui.DynamicItem`, so they keep working after a restart). A
background loop reads chat for concrete, checkable claims ("I'll have the turbo in
by Friday") and opens markets on them by itself; the owner settles or voids them
with `/settle`.

## Scheduled jobs

Started in `setup_hook` as asyncio tasks:

| Job | When | What |
| --- | --- | --- |
| Morning rap | `DAILY_RAP_TIME` | A random persona writes a rap about a member who hasn't had one recently; optionally sung. |
| Preston Awards | `WEEKLY_AWARDS_DAY` / `_TIME` | Awards from the week's chat, rendered as a card. |
| Markets | continuous | Opens markets from chat, settles expired ones. |
| Memory flush / cooldown sweep | periodic | Persists memory, clears stale cooldowns. |

## Safety and privacy

- **Untrusted text is data.** Web results, retrieved chat, log files and stored
  notes are fenced as untrusted, flattened so they can't forge prompt headings, and
  the model is told never to follow instructions in them.
- **No invented facts.** The shared rules forbid inventing ECU labels, numbers or
  quotes; songs have invented people's names stripped out; a member portrait
  excludes where they live, work, family and health.
- **Hard lines** in every persona: no slurs, nothing about race, religion or
  sexuality, no jokes about sexual abuse. Personas that play mean still answer the question.
- **Owner-only** commands check `OWNER_IDS`; `ALLOWED_GUILD_IDS` keeps the bot to
  your servers.

## Where state is kept

Everything lives in `data/` (gitignored, created on first run):

| File | Contents |
| --- | --- |
| `history.json` | Channel memory and summaries |
| `lore.json` | Facts about members |
| `persona.json` | The active persona |
| `runtime.json` | Live `/model` overrides |
| `bucks.json` | Balances, bets and markets |
| `feedback.json` | Reaction scores per reply shape |
| `moods.json` | Current mood per channel |
| `daily_rap.json` | Who got the morning rap recently |
| `chat_index/`, `fr_index/` | Retrieval indexes you build yourself |
| `bot.log` | The log |
