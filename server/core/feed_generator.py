"""
Feed video generator using FFmpeg.

Produces a short-form video (1080x1440) from a sequence of try-on images
with crossfade transitions, suitable for social media feeds.
"""

import logging
import shutil
import subprocess
from pathlib import Path

from server.core.diagnostics import PipelineLog, log_fallback

logger = logging.getLogger(__name__)


def _detect_codec() -> str:
    """Detect best available video codec: libx264 > libx265 > mpeg4."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return "libx264"
    try:
        r = subprocess.run(
            [ffmpeg, "-hide_banner", "-codecs"],
            capture_output=True, text=True, timeout=5,
        )
        for codec in ("libx264", "libx265", "mpeg4"):
            if codec in r.stdout:
                logger.info("Using video codec: %s", codec)
                return codec
    except Exception:
        pass
    return "libx264"


def build_ffmpeg_cmd(
    image_paths: list[str | Path],
    output_path: str | Path,
    duration: float = 3.0,
    fade: float = 0.5,
) -> list[str]:
    """
    Build an FFmpeg command for creating a crossfade slideshow video.

    Args:
        image_paths: ordered list of image file paths
        output_path: where to write the output MP4
        duration: how long each image is shown (seconds)
        fade: crossfade duration between images (seconds)

    Returns:
        List of command-line arguments for subprocess.
    """
    n = len(image_paths)
    if n == 0:
        return []

    codec = _detect_codec()
    cmd = ["ffmpeg", "-y"]

    # Input files
    for p in image_paths:
        cmd += ["-loop", "1", "-t", str(duration), "-i", str(p)]

    if n == 1:
        # Single image → simple video (format=yuv420p in filter, no separate -pix_fmt)
        cmd += [
            "-vf", "scale=1080:1440:force_original_aspect_ratio=decrease,pad=1080:1440:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
            "-c:v", codec,
            "-t", str(duration),
            str(output_path),
        ]
        return cmd

    # Build complex filter for crossfades
    filter_parts = []
    # Scale all inputs
    for i in range(n):
        filter_parts.append(
            f"[{i}:v]scale=1080:1440:force_original_aspect_ratio=decrease,"
            f"pad=1080:1440:(ow-iw)/2:(oh-ih)/2,format=yuv420p,setsar=1[v{i}]"
        )

    # Chain crossfades
    prev = "v0"
    offset = duration - fade
    for i in range(1, n):
        out_label = f"cf{i}" if i < n - 1 else "out"
        filter_parts.append(
            f"[{prev}][v{i}]xfade=transition=fade:duration={fade}:offset={offset}[{out_label}]"
        )
        prev = out_label
        offset += duration - fade

    filter_str = ";".join(filter_parts)
    cmd += [
        "-filter_complex", filter_str,
        "-map", "[out]",
        "-c:v", codec,
        str(output_path),
    ]
    return cmd


def generate_feed_video(
    image_paths: list[str | Path],
    output_path: str | Path,
    duration: float = 3.0,
    fade: float = 0.5,
    pipeline_log: PipelineLog | None = None,
) -> Path | None:
    """
    Generate a feed video from a sequence of try-on images.

    Args:
        image_paths: list of image paths to include in the video
        output_path: path for the output MP4 file
        duration: seconds per image
        fade: crossfade duration
        pipeline_log: optional structured log collector

    Returns:
        Path to the generated video, or None if FFmpeg is unavailable or fails.
    """
    output_path = Path(output_path)

    if not shutil.which("ffmpeg"):
        log_fallback(
            logger, "feed_video",
            FileNotFoundError("FFmpeg not found on PATH"),
            pipeline_log,
        )
        return None

    valid_paths = [Path(p) for p in image_paths if Path(p).exists()]
    if not valid_paths:
        logger.warning("No valid image paths provided for feed video")
        return None

    cmd = build_ffmpeg_cmd(valid_paths, output_path, duration, fade)
    if not cmd:
        return None

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.error("FFmpeg stderr:\n%s", result.stderr)
            log_fallback(
                logger, "feed_video",
                RuntimeError(f"FFmpeg failed (exit {result.returncode})"),
                pipeline_log,
            )
            return None

        logger.info("Feed video saved to %s", output_path)
        return output_path

    except Exception as e:
        log_fallback(logger, "feed_video", e, pipeline_log)
        return None
