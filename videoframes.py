from __future__ import annotations

import io
import logging
from typing import Iterator

import av

log = logging.getLogger("ollama-discord")

# Videos are decoded in-process with PyAV (ffmpeg library bindings, no subprocess),
# which keeps the "nothing in this project shells out" property intact.
MAX_VIDEO_BYTES = 25 * 1024 * 1024
MAX_FRAMES = 6
FRAME_MAX_EDGE = 768
JPEG_QUALITY = 80

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}


def _encode_jpeg(frame: "av.VideoFrame") -> bytes | None:
    """Downscale a decoded frame and encode it as JPEG, without PIL."""
    width, height = frame.width, frame.height
    if not width or not height:
        return None
    scale = min(1.0, FRAME_MAX_EDGE / max(width, height))
    target_w = max(16, int(width * scale)) & ~1
    target_h = max(16, int(height * scale)) & ~1
    try:
        small = frame.reformat(width=target_w, height=target_h, format="yuvj420p")
        buffer = io.BytesIO()
        with av.open(buffer, mode="w", format="mjpeg") as out:
            stream = out.add_stream("mjpeg", rate=1)
            stream.width = target_w
            stream.height = target_h
            stream.pix_fmt = "yuvj420p"
            stream.codec_context.qscale = 3
            for packet in stream.encode(small):
                out.mux(packet)
            for packet in stream.encode(None):
                out.mux(packet)
        return buffer.getvalue()
    except Exception:
        log.exception("Could not encode a frame")
        return None


def extract_frames(data: bytes, max_frames: int = MAX_FRAMES) -> tuple[list[bytes], str]:
    """Return (jpeg frames, one-line description of the clip).

    Frames are taken evenly across the whole clip so a pull is represented from
    start to finish rather than just its first moment.
    """
    if not data or len(data) > MAX_VIDEO_BYTES:
        return [], ""
    frames: list[bytes] = []
    note = ""
    try:
        with av.open(io.BytesIO(data)) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                return [], ""
            stream.thread_type = "AUTO"
            duration = float(container.duration / av.time_base) if container.duration else 0.0
            has_audio = any(s.type == "audio" for s in container.streams)
            total = stream.frames or 0

            wanted: set[int] = set()
            if total > 0:
                step = max(1, total // max_frames)
                wanted = {min(total - 1, i * step) for i in range(max_frames)}

            for index, frame in enumerate(container.decode(stream)):
                if total > 0 and index not in wanted:
                    continue
                jpeg = _encode_jpeg(frame)
                if jpeg:
                    frames.append(jpeg)
                if len(frames) >= max_frames:
                    break

            note = (
                f"{duration:.1f} second clip, {stream.width}x{stream.height}"
                + (", has an audio track you cannot hear" if has_audio else ", silent")
            )
    except Exception:
        log.exception("Could not read video")
        return [], ""
    log.info("Extracted %d frames (%s)", len(frames), note)
    return frames, note
