"""
Person image enhancer: Real-ESRGAN upscale + BiRefNet background removal.

Prepares a webcam photo for VTON by:
  1. 4x upscaling with Real-ESRGAN (identity-preserving, no face hallucination)
  2. Background removal via BiRefNet-portrait
  3. Composite on white canvas

VRAM: ~2 GB peak (ESRGAN), ~1.5 GB peak (BiRefNet). Run before FASHN VTON.
Each model is explicitly unloaded after use to free VRAM for downstream models.
"""
import logging
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


def upscale_person(img: Image.Image) -> Image.Image:
    """4x upscale with Real-ESRGAN (realesrgan-x4plus, no GFPGAN face enhancement)."""
    import torch
    import numpy as np
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        model=model,
        tile=512,       # tile processing to limit VRAM to ~2 GB
        tile_pad=10,
        pre_pad=0,
        half=True,      # fp16 for speed
        gpu_id=0 if torch.cuda.is_available() else None,
    )
    img_np = np.array(img.convert("RGB"))
    try:
        out_np, _ = upsampler.enhance(img_np, outscale=4)
        return Image.fromarray(out_np)
    finally:
        import gc
        del upsampler, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def remove_background(img: Image.Image) -> Image.Image:
    """Remove background using BiRefNet-portrait; returns RGBA."""
    import torch
    from rembg import remove, new_session
    session = new_session("birefnet-portrait")
    try:
        return remove(img, session=session)
    finally:
        import gc
        del session
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def prepare_person_for_vton(person_img: Image.Image) -> Image.Image:
    """
    Full enhancement pipeline: upscale -> remove background -> white canvas.

    Returns RGB image on white background, ready for FASHN VTON.
    """
    try:
        logger.info("Upscaling person image with Real-ESRGAN...")
        upscaled = upscale_person(person_img)
    except ImportError as e:
        logger.error("Missing package for upscaling: %s — run setup.sh", e)
        upscaled = person_img
    except Exception as e:
        logger.warning("Real-ESRGAN upscale failed (%s), using original size", e)
        upscaled = person_img

    try:
        logger.info("Removing background with BiRefNet-portrait...")
        cutout = remove_background(upscaled)  # RGBA
        canvas = Image.new("RGBA", cutout.size, (255, 255, 255, 255))
        canvas.alpha_composite(cutout)
        return canvas.convert("RGB")
    except ImportError as e:
        logger.error("Missing package for bg removal: %s — run setup.sh", e)
        return upscaled
    except Exception as e:
        logger.warning("Background removal failed (%s), using white-padded original", e)
        return upscaled
