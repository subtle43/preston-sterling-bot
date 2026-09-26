"""Image generation on the local GPU, so nothing about a picture leaves the box.

Pollinations' anonymous tier has collapsed to one small model that ignores the
prompt and stamps a watermark; Gemini's image models have no free tier at all.
This is the free option that actually works: Stable Diffusion via `diffusers`,
running inside the bot process.

The GPU is shared with CAD, so the pipeline is loaded on first use and dropped
again after a minute idle - the same idea as OLLAMA_KEEP_ALIVE. On an 8 GB
card the SDXL-class model runs with CPU offload, which keeps peak VRAM around
4 GB at the cost of a second or two per picture.

    python localimage.py "a mk7 gti on a rusty trailer"   # smoke test to data/
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
import threading
import time
import warnings

log = logging.getLogger("ollama-discord")

# The libraries are chatty: deprecation notices from diffusers, "set HF_TOKEN"
# nags from the hub, torchvision fallbacks from transformers. None of it is
# actionable and all of it lands in the bot's console, so it is quietened here
# before anything is imported.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
for _name in ("huggingface_hub", "transformers", "diffusers", "httpx", "accelerate"):
    logging.getLogger(_name).setLevel(logging.ERROR)

# DreamShaper XL Lightning: SDXL quality in 4-6 steps, single checkpoint, strong
# at illustration and caricature - which is what "draw member_a" needs.
DEFAULT_MODEL = "Lykon/dreamshaper-xl-lightning"
FP16_VAE = "madebyollin/sdxl-vae-fp16-fix"
STEPS = 8
GUIDANCE = 2.0

# Z-Image-Turbo (Alibaba, Apache-2.0): a 2025 6B-parameter model with far better
# prompt adherence than SDXL - it draws the toggle switch AND the ECU AND the
# car on jack stands. Too big for the card in bf16 (12 GB), so the transformer
# is loaded from a 5-bit GGUF (5.6 GB, resident on the GPU) and its Qwen3-4B
# text encoder stays in system RAM (8 GB) and runs once per prompt on the CPU.
# Slower than SDXL and heavier on RAM; the trade the operator asked for.
ZIMAGE_REPO = "Tongyi-MAI/Z-Image-Turbo"
ZIMAGE_GGUF = (
    "https://huggingface.co/unsloth/Z-Image-Turbo-GGUF/blob/main/z-image-turbo-Q5_K_M.gguf"
)
ZIMAGE_STEPS = 9            # the model card's recommendation is 8-9
ZIMAGE_GUIDANCE = 0.0       # distilled: no classifier-free guidance
ZIMAGE_WIDTH, ZIMAGE_HEIGHT = 1152, 640


def is_zimage(model: str) -> bool:
    return "z-image" in (model or "").lower()
# Unload after this long idle so CAD gets the VRAM back.
IDLE_UNLOAD = 2 * 60
# Where the parts live. The UNet and VAE run every step and sit on the GPU;
# the two CLIP text encoders run once per prompt and stay in system RAM. On
# the 8 GB card that is ~5.3 GB resident plus ~0.7 GB working, leaving room
# for Fusion's ~1.3 GB. Full CPU offload (everything in RAM, stages shuttled
# in and out) is the fallback when the card has less free than this.
RESIDENT_NEEDS_MB = 6_500

NEGATIVE = (
    "text, watermark, logo, caption, signature, blurry, deformed, extra limbs, "
    "extra fingers, low quality, jpeg artifacts"
)

_pipe = None
_pipe_model = ""
_resident = False
# The two CLIP encoders, detached from the pipeline and kept in fp32 on the CPU.
# Detached because the pipeline force-casts supplied embeddings to the dtype of
# whatever encoder it holds; with none held it uses the UNet's fp16 and stays
# out of the way.
_encoders: tuple | None = None
# The img2img and inpaint pipelines, built lazily from the loaded Z-Image
# components - they share the transformer, VAE and encoders, so no extra VRAM.
_edit_pipe = None
_inpaint_pipe = None
_last_used = 0.0
# ONE lock for the whole card, shared with songgen.py: the image and song
# models evict each other, and two locks taken in opposite orders (song thread
# holding its lock -> localimage.unload(); image thread holding this lock ->
# songgen.unload()) could deadlock both. Re-entrant, because a load evicts the
# other model while already holding it.
_lock = threading.RLock()
_load_error = ""


def available() -> bool:
    try:
        import torch  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


def _load(model: str):
    """Build the pipeline once. Called under _lock, off the event loop."""
    global _pipe, _pipe_model, _load_error
    if _pipe is not None and _pipe_model == model:
        return _pipe
    if is_zimage(model):
        return _load_zimage(model)
    _evict_song_model()
    import torch
    from diffusers import AutoencoderKL, AutoPipelineForText2Image, DPMSolverMultistepScheduler

    t0 = time.monotonic()
    pipe = AutoPipelineForText2Image.from_pretrained(
        model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    # The stock SDXL VAE overflows in fp16, so diffusers decodes it in fp32 -
    # a 2.3 GB spike that pushed peak VRAM to 7.5 GB and spilled into shared
    # memory. This VAE was retrained to be fp16-safe: same output, no spike.
    try:
        pipe.vae = AutoencoderKL.from_pretrained(FP16_VAE, torch_dtype=torch.float16)
    except Exception:
        log.warning("fp16-fix VAE unavailable - using the stock one (slower decode)")
    # Lightning / turbo merges want a low-step SDE sampler.
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"
    )
    free_mb, total_mb = (x // (1024 * 1024) for x in torch.cuda.mem_get_info())
    global _resident
    _resident = free_mb >= RESIDENT_NEEDS_MB
    if _resident:
        pipe.unet.to("cuda")
        pipe.vae.to("cuda")
        # fp32 on the CPU: fp16 CLIP has no fast CPU kernels and took 3.4 s per
        # prompt; fp32 takes 0.6 s for ~1.6 GB more RAM. Worth it.
        global _encoders
        _encoders = (
            pipe.tokenizer, pipe.text_encoder.to("cpu", torch.float32),
            pipe.tokenizer_2, pipe.text_encoder_2.to("cpu", torch.float32),
        )
        pipe.text_encoder = None
        pipe.text_encoder_2 = None
        pipe.vae.enable_slicing()
    else:
        pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    _pipe, _pipe_model, _load_error = pipe, model, ""
    log.info("Local image model %s loaded in %.1fs (%d/%d MB free, %s)",
             model, time.monotonic() - t0, free_mb, total_mb,
             "unet resident on GPU" if _resident else "cpu offload")
    return pipe


def _evict_song_model() -> None:
    """The song model and this one do not both fit on the card.

    Both modules share the one re-entrant GPU lock (see _lock), so this is a
    plain nested acquire, not a cross-lock.
    """
    try:
        import songgen
        songgen.unload()
    except Exception:
        pass


def _load_zimage(model: str):
    global _pipe, _pipe_model, _load_error, _resident, _encoders
    _evict_song_model()
    import torch
    from diffusers import GGUFQuantizationConfig, ZImagePipeline, ZImageTransformer2DModel

    t0 = time.monotonic()
    transformer = ZImageTransformer2DModel.from_single_file(
        ZIMAGE_GGUF,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
    )
    pipe = ZImagePipeline.from_pretrained(
        ZIMAGE_REPO, transformer=transformer, torch_dtype=torch.bfloat16
    )
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    # Detached, same reason as the SDXL encoders: the pipeline's idea of its own
    # device comes from the modules it holds, and it must think "cuda".
    _encoders = (pipe.tokenizer, pipe.text_encoder.to("cpu", torch.bfloat16))
    pipe.text_encoder = None
    pipe.set_progress_bar_config(disable=True)
    _resident = True
    _pipe, _pipe_model, _load_error = pipe, model, ""
    free_mb = torch.cuda.mem_get_info()[0] // (1024 * 1024)
    log.info("Local image model %s loaded in %.1fs (Q5 GGUF on GPU, %d MB free after)",
             model, time.monotonic() - t0, free_mb)
    return pipe


def _fit(image, long_side: int = 1152):
    """Resize to at most `long_side`, dimensions a multiple of 16, RGB."""
    from PIL import Image
    image = image.convert("RGB")
    w, h = image.size
    scale = min(1.0, long_side / max(w, h))
    w, h = max(16, int(w * scale) // 16 * 16), max(16, int(h * scale) // 16 * 16)
    return image.resize((w, h), Image.LANCZOS)


def _edit_sync(prompt: str, image_bytes: bytes, strength: float, seed: int | None) -> bytes:
    """Image-to-image with Z-Image: keep the photo's composition, change what
    the prompt says. `strength` is how far to depart from the original -
    0.35 touches it up, 0.6 restyles it, 0.75 changes what is in it."""
    global _last_used, _edit_pipe
    import math
    import torch
    from PIL import Image
    from diffusers import ZImageImg2ImgPipeline
    with _lock:
        _last_used = time.monotonic()
        pipe = _load(ZIMAGE_REPO)
        if _edit_pipe is None:
            _edit_pipe = ZImageImg2ImgPipeline(
                scheduler=pipe.scheduler, vae=pipe.vae, text_encoder=None,
                tokenizer=pipe.tokenizer, transformer=pipe.transformer,
            )
            _edit_pipe.set_progress_bar_config(disable=True)
        src = _fit(Image.open(io.BytesIO(image_bytes)))
        gen = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
        t0 = time.monotonic()
        embeds = _encode_zimage(prompt)
        # img2img only runs the last `strength` share of the schedule, so scale
        # the step count up to keep ~9 real denoising steps.
        steps = max(ZIMAGE_STEPS, math.ceil(ZIMAGE_STEPS / max(strength, 0.2)))
        image = _edit_pipe(
            prompt_embeds=embeds, image=src, strength=strength,
            num_inference_steps=steps, guidance_scale=ZIMAGE_GUIDANCE,
            width=src.width, height=src.height, generator=gen,
        ).images[0]
        _last_used = time.monotonic()
        log.info("Local image edit: %dx%d strength %.2f in %.1fs for %r",
                 src.width, src.height, strength, _last_used - t0, prompt[:60])
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _inpaint_sync(
    prompt: str, image_bytes: bytes, box: tuple[float, float, float, float],
    grow: float, seed: int | None,
) -> bytes:
    """Repaint ONLY a region. `box` is (x0, y0, x1, y1) as fractions of the
    image; it is grown by `grow` of its own size so the effect has room, the
    mask edge is feathered, and the original pixels are pasted back outside it
    so nothing else in the photo moves by a single pixel."""
    global _last_used, _inpaint_pipe
    import torch
    from PIL import Image, ImageDraw, ImageFilter
    from diffusers import ZImageInpaintPipeline
    with _lock:
        _last_used = time.monotonic()
        pipe = _load(ZIMAGE_REPO)
        if _inpaint_pipe is None:
            _inpaint_pipe = ZImageInpaintPipeline(
                scheduler=pipe.scheduler, vae=pipe.vae, text_encoder=None,
                tokenizer=pipe.tokenizer, transformer=pipe.transformer,
            )
            _inpaint_pipe.set_progress_bar_config(disable=True)
        src = _fit(Image.open(io.BytesIO(image_bytes)))
        W, H = src.size
        x0, y0, x1, y1 = box
        bw, bh = max(x1 - x0, 0.02), max(y1 - y0, 0.02)
        gx, gy = bw * grow, bh * grow
        # At least a decent patch: a tiny exhaust tip needs more than itself.
        gx, gy = max(gx, 0.10), max(gy, 0.10)
        px0, py0 = int(max(0.0, x0 - gx) * W), int(max(0.0, y0 - gy) * H)
        px1, py1 = int(min(1.0, x1 + gx) * W), int(min(1.0, y1 + gy) * H)
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).rectangle([px0, py0, px1, py1], fill=255)
        feather = max(8, int(min(W, H) * 0.02))
        soft = mask.filter(ImageFilter.GaussianBlur(feather))
        gen = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
        t0 = time.monotonic()
        embeds = _encode_zimage(prompt)
        out = _inpaint_pipe(
            prompt_embeds=embeds, image=src, mask_image=mask, strength=1.0,
            num_inference_steps=ZIMAGE_STEPS, guidance_scale=ZIMAGE_GUIDANCE,
            width=W, height=H, generator=gen,
        ).images[0].convert("RGB")
        # Pixel-exact outside the (feathered) mask.
        result = Image.composite(out, src, soft)
        _last_used = time.monotonic()
        log.info("Local inpaint: box %dx%d of %dx%d in %.1fs for %r",
                 px1 - px0, py1 - py0, W, H, _last_used - t0, prompt[:60])
    buf = io.BytesIO()
    result.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


async def inpaint(
    prompt: str, image_bytes: bytes, box: tuple[float, float, float, float], *,
    grow: float = 0.6, seed: int | None = None,
) -> bytes | None:
    global _load_error
    if not prompt or not image_bytes or not available() or _load_error:
        return None
    try:
        return await asyncio.to_thread(_inpaint_sync, prompt, image_bytes, box, grow, seed)
    except Exception as exc:
        _load_error = f"{type(exc).__name__}: {exc}"[:200]
        log.exception("Local inpaint failed - disabled until restart")
        return None


async def edit(
    prompt: str, image_bytes: bytes, *, strength: float = 0.6, seed: int | None = None,
) -> bytes | None:
    """Edit a picture. Z-Image only; None if unavailable."""
    global _load_error
    if not prompt or not image_bytes or not available() or _load_error:
        return None
    try:
        return await asyncio.to_thread(_edit_sync, prompt, image_bytes, strength, seed)
    except Exception as exc:
        _load_error = f"{type(exc).__name__}: {exc}"[:200]
        log.exception("Local image edit failed - disabled until restart")
        return None


def _encode_zimage(text: str):
    """Z-Image's prompt encoding on the CPU: Qwen3 chat template, penultimate
    hidden layer, padding stripped - the pipeline's own recipe."""
    import torch
    tokenizer, encoder = _encoders
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )
    inputs = tokenizer(
        [chat], padding="max_length", max_length=512, truncation=True, return_tensors="pt"
    )
    mask = inputs.attention_mask.bool()
    with torch.no_grad():
        hidden = encoder(
            input_ids=inputs.input_ids, attention_mask=mask, output_hidden_states=True
        ).hidden_states[-2]
    return [hidden[0][mask[0]].to("cuda", torch.bfloat16)]


def _encode(text: str):
    """SDXL's dual-CLIP prompt encoding, done by hand on the CPU in fp32.

    Same recipe as the pipeline's encode_prompt: penultimate hidden layer from
    each encoder, concatenated; the pooled projection from the second one.
    """
    import torch
    tok, enc, tok2, enc2 = _encoders
    embeds = []
    pooled = None
    with torch.no_grad():
        for tokenizer, encoder in ((tok, enc), (tok2, enc2)):
            ids = tokenizer(
                text, padding="max_length", max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids
            out = encoder(ids, output_hidden_states=True)
            pooled = out[0]                      # the last one wins: encoder 2's projection
            embeds.append(out.hidden_states[-2])
    return torch.cat(embeds, dim=-1), pooled


def clear_error() -> None:
    """Let a different model be tried after one failed to load."""
    global _load_error
    _load_error = ""


def unload() -> None:
    global _pipe, _pipe_model, _encoders
    with _lock:
        if _pipe is None:
            return
        _pipe, _pipe_model, _encoders = None, "", None
        global _edit_pipe, _inpaint_pipe
        _edit_pipe = None
        _inpaint_pipe = None
        try:
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
        log.info("Local image model unloaded (idle)")


def _generate_sync(
    prompt: str, model: str, width: int, height: int, seed: int | None, square: bool = False,
) -> bytes:
    global _last_used
    import torch
    with _lock:
        # Touched BEFORE loading and rendering, not just after. Left at zero the
        # unloader saw "idle since forever" during the first render, queued an
        # unload behind the lock, and dropped the model the second it finished
        # - so every picture paid a full reload.
        _last_used = time.monotonic()
        pipe = _load(model)
        gen = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
        t0 = time.monotonic()
        if is_zimage(model):
            t_enc = time.monotonic()
            embeds = _encode_zimage(prompt)
            log.info("Z-Image prompt encoded on CPU in %.1fs", time.monotonic() - t_enc)
            # Album covers are square; everything else is the widescreen default.
            zw, zh = (1024, 1024) if square else (ZIMAGE_WIDTH, ZIMAGE_HEIGHT)
            image = pipe(
                prompt_embeds=embeds, num_inference_steps=ZIMAGE_STEPS,
                guidance_scale=ZIMAGE_GUIDANCE, width=zw, height=zh,
                generator=gen,
            ).images[0]
            _last_used = time.monotonic()
            log.info("Local image: %dx%d in %.1fs for %r", zw, zh,
                     _last_used - t0, prompt[:60])
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=92)
            return buf.getvalue()
        if square:
            width = height = 1024
        kwargs = dict(
            num_inference_steps=STEPS, guidance_scale=GUIDANCE,
            width=width, height=height, generator=gen,
        )
        if _resident:
            # The pipeline assumes one device. With the encoders on the CPU and
            # the UNet on the GPU, encode here and hand the embeddings across.
            pe, ppe = _encode(prompt)
            npe, nppe = _encode(NEGATIVE)
            half = torch.float16
            # Latents made here in fp16 as well: with fp32 encoders registered,
            # the pipeline's own idea of its dtype is fp32 and it would build
            # fp32 latents for an fp16 UNet.
            latents = torch.randn(
                (1, pipe.unet.config.in_channels, height // 8, width // 8),
                generator=gen, device="cuda", dtype=half,
            )
            kwargs.update(
                prompt_embeds=pe.to("cuda", half), negative_prompt_embeds=npe.to("cuda", half),
                pooled_prompt_embeds=ppe.to("cuda", half),
                negative_pooled_prompt_embeds=nppe.to("cuda", half),
                latents=latents,
            )
        else:
            kwargs.update(prompt=prompt, negative_prompt=NEGATIVE)
        image = pipe(**kwargs).images[0]
        _last_used = time.monotonic()
        log.info("Local image: %dx%d in %.1fs for %r", width, height, _last_used - t0, prompt[:60])
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


async def generate(
    prompt: str, *, model: str = DEFAULT_MODEL, width: int = 1024, height: int = 576,
    seed: int | None = None, square: bool = False,
) -> bytes | None:
    """One picture, or None if the GPU path is unavailable or failed."""
    global _load_error
    if not prompt or not available():
        return None
    if _load_error:
        return None                      # do not retry a broken install per picture
    try:
        return await asyncio.to_thread(_generate_sync, prompt, model, width, height, seed, square)
    except Exception as exc:
        _load_error = f"{type(exc).__name__}: {exc}"[:200]
        log.exception("Local image generation failed - disabled until restart")
        return None


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
        description="Render an image locally. The prompt goes to the model as typed - "
                    "none of the bot's filtering, persona or cooldowns apply here."
    )
    ap.add_argument("prompt", nargs="+", help="what to draw")
    ap.add_argument("-o", "--out", default=None, help="output file (default data/img_<time>.jpg)")
    ap.add_argument("-m", "--model", default="z", help="'z' (Z-Image-Turbo), 'sdxl' (DreamShaper), or a HF repo id")
    ap.add_argument("-s", "--seed", type=int, default=None, help="seed for a repeatable result")
    ap.add_argument("-n", "--count", type=int, default=1, help="how many variations")
    ap.add_argument("-W", "--width", type=int, default=None)
    ap.add_argument("-H", "--height", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model = {"z": ZIMAGE_REPO, "sdxl": DEFAULT_MODEL}.get(args.model.lower(), args.model)
    if args.width and args.height and is_zimage(model):
        ZIMAGE_WIDTH, ZIMAGE_HEIGHT = args.width, args.height
    text = " ".join(args.prompt)

    async def _run() -> None:
        for i in range(args.count):
            seed = (args.seed + i) if args.seed is not None else None
            kw = {}
            if args.width and args.height and not is_zimage(model):
                kw = {"width": args.width, "height": args.height}
            data = await generate(text, model=model, seed=seed, **kw)
            if not data:
                print("failed:", _load_error or "unavailable")
                return
            out = Path(args.out) if args.out else Path("data") / f"img_{int(time.time())}_{i}.jpg"
            if args.count > 1 and args.out:
                out = out.with_name(f"{out.stem}_{i}{out.suffix}")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            print(f"wrote {out} ({len(data):,} bytes)")

    asyncio.run(_run())
