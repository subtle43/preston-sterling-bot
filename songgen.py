"""Song generation on the local GPU: lyrics in, a sung clip out.

ACE-Step 1.5 (Apache-2.0) is the open Suno-alike: give it a style caption and
tagged lyrics and it sings them. The XL-turbo build runs in 8 steps and, with
CPU offload, peaks around 4 GB of VRAM - which on an 8 GB card means it cannot
share the GPU with the image model. Loading one evicts the other, both ways.

Same lifecycle as localimage.py: loaded on first use, dropped after two minutes
idle, disabled until restart if it fails to load.

    python songgen.py "country rap, male vocals, banjo, 110 bpm" --lyrics l.txt
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import threading
import time
import warnings

log = logging.getLogger("ollama-discord")

# Same quietening as localimage.py; setdefault, so whichever imports first wins
# and the second is a no-op.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
for _name in ("huggingface_hub", "transformers", "diffusers", "httpx", "accelerate"):
    logging.getLogger(_name).setLevel(logging.ERROR)

DEFAULT_MODEL = "ACE-Step/acestep-v15-xl-turbo-diffusers"
STEPS = 8
DURATION = 30.0
GUIDANCE = 1.0          # turbo build: CFG is ignored, 1.0 keeps the pipeline quiet
SHIFT = 3.0
IDLE_UNLOAD = 2 * 60
MP3_BITRATE = 160_000
# It sings about thirty characters a second. Anything past that gets sung at
# auctioneer speed or silently dropped, so cap what reaches the model by clip.
LYRIC_CHARS_PER_SEC = 32

_pipe = None
_pipe_model = ""
_last_used = 0.0
# The image model's lock IS this module's lock: one card, one queue. See the
# note on localimage._lock for why it must be shared and re-entrant.
import localimage as _localimage  # noqa: E402  - module-level state only, torch stays lazy
_lock = _localimage._lock
_load_error = ""
_force_wav = False       # smoke-test switch for the WAV fallback


def available() -> bool:
    try:
        import torch
        import diffusers.pipelines.ace_step  # noqa: F401  - needs diffusers >= 0.40
        return torch.cuda.is_available()
    except Exception:
        return False


def _load(model: str):
    """Build the pipeline once. Called under _lock, off the event loop."""
    global _pipe, _pipe_model
    if _pipe is not None and _pipe_model == model:
        return _pipe
    import torch
    from diffusers import AceStepPipeline

    # The image model and this one do not fit together. Same lock, so this
    # is a nested acquire, not a cross-lock.
    try:
        _localimage.unload()
    except Exception:
        pass

    t0 = time.monotonic()
    pipe = AceStepPipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
    # Stages shuttle on and off the card as they run: the text encoder, then
    # the DiT, then the VAE. Peak is the DiT alone, not the sum.
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    _pipe, _pipe_model = pipe, model
    try:
        free, total = torch.cuda.mem_get_info()
        log.info("Song model %s loaded in %.1fs (cpu offload, %d MB free)",
                 model, time.monotonic() - t0, free // 2**20)
    except Exception:
        log.info("Song model %s loaded in %.1fs", model, time.monotonic() - t0)
    return pipe


def _generate_sync(
    caption: str, lyrics: str, model: str, duration: float, steps: int,
    seed: int | None, bpm: int | None,
) -> tuple[bytes, str]:
    global _last_used
    import torch
    with _lock:
        # Touched BEFORE loading and rendering - see localimage._generate_sync.
        _last_used = time.monotonic()
        pipe = _load(model)
        gen = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
        t0 = time.monotonic()
        out = pipe(
            prompt=caption,
            lyrics=lyrics[:int(duration * LYRIC_CHARS_PER_SEC)],
            audio_duration=float(duration),
            vocal_language="en",
            num_inference_steps=steps,
            guidance_scale=GUIDANCE,
            shift=SHIFT,
            bpm=bpm,
            generator=gen,
            output_type="pt",
        ).audios
        rate = int(getattr(pipe, "sample_rate", 48_000))
        # (batch, channels, samples) -> (channels, samples) float32 in [-1, 1]
        audio = out[0].float().clamp(-1, 1).cpu().numpy()
        _last_used = time.monotonic()
        log.info("Local song: %.0fs of audio in %.1fs for %r",
                 audio.shape[1] / rate, _last_used - t0, caption[:60])
    # Encoding is CPU work and needs no GPU; do it outside the lock.
    try:
        if _force_wav:
            raise RuntimeError("WAV forced")
        return encode_mp3(audio, rate), "mp3"
    except Exception:
        log.exception("MP3 encode failed - sending WAV")
        return encode_wav(audio, rate), "wav"


def encode_mp3(audio, rate: int) -> bytes:
    """(2, N) float32 -> MP3 bytes, in-process through PyAV's bundled LAME."""
    import av
    import numpy as np

    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp3") as out:
        stream = out.add_stream("libmp3lame", rate=rate)
        stream.layout = "stereo"
        stream.format = "fltp"
        stream.bit_rate = MP3_BITRATE
        frame_len = 1152                     # one MPEG audio frame
        pts = 0
        for start in range(0, audio.shape[1], frame_len):
            chunk = np.ascontiguousarray(audio[:, start:start + frame_len], dtype=np.float32)
            frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout="stereo")
            frame.sample_rate = rate
            frame.pts = pts
            pts += chunk.shape[1]
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return buf.getvalue()


def encode_wav(audio, rate: int) -> bytes:
    """Fallback: 16-bit stereo WAV from the standard library. ~11.5 MB a minute - over Discord's 10 MB cap past ~50 s, which the caller checks."""
    import wave
    import numpy as np

    pcm = (np.clip(audio, -1, 1).T * 32767).astype("<i2")   # interleave L/R
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()


async def generate(
    caption: str, lyrics: str, *, model: str = DEFAULT_MODEL, duration: float = DURATION,
    steps: int = STEPS, seed: int | None = None, bpm: int | None = None,
) -> tuple[bytes, str] | None:
    """One clip as (bytes, "mp3"|"wav"), or None if the GPU path is unavailable or failed."""
    global _load_error
    if not caption or not available():
        return None
    if _load_error:
        return None                      # do not retry a broken install per song
    try:
        return await asyncio.to_thread(
            _generate_sync, caption, lyrics or "", model, duration, steps, seed, bpm
        )
    except Exception as exc:
        _load_error = f"{type(exc).__name__}: {exc}"[:200]
        log.exception("Local song generation failed - disabled until restart")
        return None


def clear_error() -> None:
    global _load_error
    _load_error = ""


def unload() -> None:
    global _pipe, _pipe_model
    with _lock:
        if _pipe is None:
            return
        _pipe, _pipe_model = None, ""
        try:
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
        log.info("Song model unloaded")


async def idle_unloader() -> None:
    """Background task: drop the model after IDLE_UNLOAD seconds unused."""
    while True:
        await asyncio.sleep(10)
        if _pipe is not None and time.monotonic() - _last_used > IDLE_UNLOAD:
            await asyncio.to_thread(unload)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(
        description="Sing a clip locally. The caption and lyrics go to the model as "
                    "given - none of the bot's persona, filtering or cooldowns apply."
    )
    ap.add_argument("caption", nargs="+", help="style tags: genre, mood, instruments, vocals, bpm")
    ap.add_argument("-l", "--lyrics", default=None, help="text file with [verse]/[chorus] tagged lyrics")
    ap.add_argument("-o", "--out", default=None, help="output file (default data/song_<time>.mp3)")
    ap.add_argument("-d", "--duration", type=float, default=DURATION)
    ap.add_argument("-s", "--seed", type=int, default=None)
    ap.add_argument("-b", "--bpm", type=int, default=None)
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    ap.add_argument("--wav", action="store_true", help="skip MP3 and write WAV")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    caption = " ".join(args.caption)
    lyrics = Path(args.lyrics).read_text(encoding="utf-8") if args.lyrics else (
        "[verse]\nLeft the boost gauge pinned again\n"
        "Told the room the map was fine\n"
        "Every log a flat line\n"
        "Every fix a friend of mine\n\n"
        "[chorus]\nRun it hot, run it loud\n"
        "Blame the tune, blame the crowd\n"
        "Run it hot, run it loud\n"
        "Never once was it the gas\n"
    )
    print("First run downloads several GB from huggingface.co - allow a few minutes.")
    _force_wav = args.wav

    async def _run() -> None:
        result = await generate(
            caption, lyrics, model=args.model, duration=args.duration,
            seed=args.seed, bpm=args.bpm,
        )
        if not result:
            print("failed:", _load_error or "unavailable")
            return
        data, ext = result
        out = Path(args.out) if args.out else Path("data") / f"song_{int(time.time())}.{ext}"
        if out.suffix.lstrip(".") != ext:
            out = out.with_suffix("." + ext)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        print(f"wrote {out} ({len(data):,} bytes)")

    asyncio.run(_run())
