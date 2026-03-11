"""
AI person regeneration: webcam photo -> neutral-pose figure on white background.

Uses SD1.5 + ControlNet (OpenPose) + IP-Adapter (FaceID Plus v2) to generate
a new high-quality image of the person in a neutral standing pose.

VRAM: ~10 GB peak. Run BEFORE FASHN VTON (unload before loading VTON).
"""
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

NEUTRAL_POSE_PATH = Path("client/assets/neutral_pose_skeleton.png")


def regenerate_person(
    webcam_img: Image.Image,
    output_path: Path | None = None,
) -> Image.Image | None:
    """
    Regenerate person in neutral pose on white background.
    Returns regenerated PIL Image, or None on failure.
    """
    try:
        # 1. Extract face embedding
        logger.info("Extracting face embedding with InsightFace...")
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        faces = app.get(np.array(webcam_img.convert("RGB"))[:, :, ::-1])  # RGB->BGR
        if not faces:
            logger.warning("No face detected — cannot regenerate")
            return None
        face_embed = torch.tensor(faces[0].normed_embedding).unsqueeze(0)
        del app
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. Load pipeline
        logger.info("Loading SD1.5 + ControlNet + IP-Adapter...")
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16,
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipe.to("cuda")
        pipe.load_ip_adapter(
            "h94/IP-Adapter-FaceID",
            subfolder=None,
            weight_name="ip-adapter-faceid-plusv2_sd15.bin",
        )
        pipe.set_ip_adapter_scale(0.6)

        # 3. Load neutral pose skeleton
        pose_img = Image.open(NEUTRAL_POSE_PATH).resize((512, 768))

        # 4. Generate
        logger.info("Generating neutral-pose person image...")
        result = pipe(
            prompt=(
                "full body photo of a person, standing straight, arms at sides, "
                "plain white t-shirt, jeans, white background, "
                "professional studio photography, high quality, sharp"
            ),
            negative_prompt=(
                "blurry, low quality, distorted, deformed, extra limbs, "
                "cropped, watermark"
            ),
            image=pose_img,
            ip_adapter_image_embeds=[face_embed],
            num_inference_steps=30,
            guidance_scale=7.5,
            width=512,
            height=768,
        ).images[0]

        # 5. Cleanup
        del pipe, controlnet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(str(output_path))
            logger.info("Regenerated person saved to %s", output_path)

        return result

    except ImportError as e:
        logger.error("Missing package for regeneration: %s", e)
        return None
    except Exception as e:
        logger.error("Person regeneration failed: %s", e)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None
