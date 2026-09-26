# Preston's personalities

Each folder here is one personality. Switch live in Discord with `/persona` - anyone can, with a 5-minute cooldown (owner exempt), no restart.

Current: therapist, gary, narrator, linkedin, butler, salesman, preacher, car, sportscaster,
wizard, hr, victorian, peasant, conspiracy, mean, classic, roaster, trump,
chef, boomer, zoomer, noir, sergeant, cat, grandma, sommelier, mafia, pilot,
pirate, shakespeare, anime, surfer, auctioneer, influencer, lawyer, weather, alien, villain,
drunkard, freebaser, idiot.

## Files in a persona folder

| File | What reads it |
|---|---|
| `system.txt` | The full harness (normal chat on Gemini or `/model harness:full`). Only the CHARACTER goes here - see below. |
| `lite.txt` | The lite harness, and the voice of every slash-command gag (`/dyno`, `/sue`, `/card`, Preston Bucks, `/wordle`...). A short paragraph or two. |
| `voice_note.txt` | One paragraph attached to the END of every reply. The model obeys what it reads last, so this is what keeps the voice on track. |
| `about.txt` | One short line shown in the `/persona` list and searched when typing (e.g. "drunk" finds drunkard). |
| `song_style.txt` | Optional. Music genres for songs and raps, one "genre: description" per line (one is picked per song). Without it, songs use a random style. |
| `image_style.txt` | Optional. The default look for pictures when the request names no style (e.g. noir = black-and-white film noir). Without it, photorealistic. |
| `flags.txt` | Optional switches, one per line: `big_words` (gets the weekly list of precise words; for erudite characters only - a salesman given "thixotropic" will use it) and `no_drunk` (never plays the drunk mood). |

## The shared rules

`_shared/rules.txt` holds everything that makes Preston useful and safe whatever his personality:
never refuse, answer first, length, knowing his stuff, never inventing ECU labels or numbers,
privacy, moods, drunk mode, what he can do. It is attached automatically after any `system.txt`
that does not contain the line `THE RULE ABOVE ALL OTHERS` - so a persona file only needs the
character. (mean, conspiracy and classic still carry their own full copy of the rules.)
Edit the shared file once and every new-style persona picks it up.

## Adding a new personality

1. Copy a folder, e.g. `gary` -> `pirate`.
2. Rewrite `system.txt`: who he is, THE BIT with an example or two, HOW THE BIT WORKS, and keep
   the "THE ANSWER IS REAL" and "HARD LINES" paragraphs at the end.
3. Rewrite `lite.txt` and `voice_note.txt` in the new voice (keep the hard lines in both),
   and write a one-line `about.txt`.
4. `/persona pirate` in Discord. Files are read fresh, so edits apply on the next reply.

Tips:
- Example lines get copied word for word. Say they are style samples only (the "THE ANSWER IS
  REAL" paragraph in the existing files does this).
- Hard lines in every persona: no slurs; nothing about race, religion or sexuality; no jokes
  about rape or abuse.

## Files outside this folder

`prompts/system.txt` and `prompts/lite.txt` are only fallbacks, used if no persona folder
exists. Edit the persona folders, not those. `prompts/voice.txt` is the prompt for the
trained voice model and is separate from personas.
