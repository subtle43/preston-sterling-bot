# Commands

Three ways to use Preston: slash commands, `!` prefix commands, and plain
language to the bot (@mention it or reply to it). "Owner" means a user ID listed in
`OWNER_IDS`.

## Slash commands

### Talking

| Command | Who | What it does |
| --- | --- | --- |
| `/ask prompt:` | Everyone | Ask Preston anything. |
| `/summarize` | Everyone | Recap recent messages in this channel, optionally for one topic or person. (`/summerize` works too.) |
| `/fr query:` | Everyone | Look something up in the Funktionsrahmen (needs an FR index, see [RAG.md](RAG.md)). |
| `/persona [name]` | Everyone | Switch personality. Empty shows the list with descriptions; typing a word (drunk, cat, pirate) searches; `random` picks one. 5-minute cooldown between switches (owner exempt); each switch is announced. |

### Gags

All of these build a dossier of the member from the chat archive and lore, and
are written in the current persona's voice.

| Command | What it does |
| --- | --- |
| `/dyno member:` | A fake chassis-dyno sheet for their car, with their real problems as dips in the curve. |
| `/sue defendant: crime:` | Tuning Court: a trial with prosecution, the defendant's own words as evidence, and a verdict. |
| `/race left: right:` | A quarter-mile drag race between two members' cars, with a time slip and a race call. |
| `/tierlist topic:` | Ranks the server's regulars S to F on anything. |
| `/stock member:` | A member's claimed horsepower, charted like a stock with financial-news commentary. |
| `/factcheck member:` | A cable-news fact check of a member against their own words. |
| `/card member:` | A trading card: title, type, HP, two moves, weakness, flavour text, rarity. |
| `/wordle` | Preston plays today's Wordle honestly and posts a spoiler-free grid. |
| `/awards` | Owner: hand out this week's Preston Awards now (they also run on a schedule). |

### Preston Bucks

| Command | Who | What it does |
| --- | --- | --- |
| `/bucks` | Everyone | Your balance. |
| `/daily` | Everyone | Claim your daily Bucks. |
| `/leaderboard` | Everyone | Richest and brokest. |
| `/markets` | Everyone | Open markets with odds and bet buttons. |
| `/bet market: option: amount:` | Everyone | Bet on an open market (or use the buttons under a market). |
| `/odds question: [hours]` | Everyone | Ask the bookie to open a market on anything ("member_x's car runs by Friday"). |
| `/settle` | Owner | Settle or void a market. |

The bookie also opens markets by himself when someone in chat makes a concrete,
checkable claim.

### Admin

| Command | Who | What it does |
| --- | --- | --- |
| `/model` | Owner | Switch model, backend (Gemini/Ollama) and harness live. Saved across restarts. |
| `/dm user: message:` | Owner | Have the bot DM someone in the server (disabled if `OWNER_IDS` is empty). |

## Prefix commands

| Command | Who | What it does |
| --- | --- | --- |
| `!ai <message>` | Everyone | Talk to Preston without an @mention (`COMMAND_PREFIX`). |
| `!summarize`, `!tldr`, `!sum` | Everyone | Summarise the channel. |
| `!car <your car>` | Everyone | Tell Preston what you drive: `!car MK7 GTI, IS38, E30`. `!car` alone shows it. |
| `!remember <fact>` | Everyone | Store a short note about yourself: `!remember I run 93 and a stage 2 tune`. |
| `!whois [@member]` | Everyone | What Preston knows about someone. |
| `!forget` | Everyone | Delete everything Preston has stored about you. |
| `!memory` | Everyone | What the bot is holding for this channel: live room, kept messages, summary. |
| `!clear` | Owner | Wipe this channel's conversation memory. |
| `!compact` | Owner | Fold this channel's memory into the summary now. |
| `!mood [name]` | Owner | Show or set the channel's mood. |
| `!rap` / `!rap here` / `!rap sing` | Owner | Post the morning rap now, post it here, or record today's rap as a song. |
| `!feedback` | Owner | The reaction scores per reply shape. |
| `!why` | Owner | The decision trail behind the last reply in this channel. |
| `!nick @member name` | Owner | Give a member a nickname the bot uses. |

## Plain language

Say these to Preston (@mention or reply) and the intent classifier routes them:

| Say | What happens |
| --- | --- |
| "draw / render / paint ..." | A picture, in the persona's house style unless you name one. |
| "make a meme of ..." | A captioned meme. Under someone's message, it riffs on their words. |
| "shittify this car", "make it red" (on a photo) | Edits the photo. |
| "make it darker", "again but as a cartoon" | Redoes the last picture (30-minute window). |
| "make a song / rap / diss track / poem about @member" | Lyrics from their history. Songs are sung (MP3 plus cover art) unless you say "just the lyrics". |
| "... as a sea shanty", "... in the style of Johnny Cash" | Sets the genre for the words and the music. |
| "make a song about <thing>" | A song about a server topic, built from the archive. |
| attach a `.csv` log + "review this" | Full datalog review with a chart. |
| "what did @member say about ..." | Searches the chat archive. |
| "look up ..." / anything current | Web search, if enabled. |
| "draw his car and write a song about it" | Both, in order. |
