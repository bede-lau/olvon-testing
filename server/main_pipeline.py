"""
Orchestrator pipeline: body measurements -> virtual try-on -> sizing -> feed video.

2D VTON pipeline using FASHN VTON v1.5 for photorealistic virtual try-on.
"""

import argparse
import json
import logging
from pathlib import Path

from server.core.body_measurements import extract as extract_measurements
from server.core.diagnostics import PipelineLog
from server.core.feed_generator import generate_feed_video
from server.core.sizing_logic import recommend_size
from server.core.tryon_worker import TryOnWorker

logger = logging.getLogger(__name__)


class VTONPipeline:
    """End-to-end pipeline: measurements -> try-on -> sizing -> feed video."""

    def __init__(
        self,
        output_dir: str | Path = "server/outputs",
        weights_dir: str | Path = "server/lib/fashn-vton/weights",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline_log = PipelineLog()
        self.tryon = TryOnWorker(weights_dir)

    def run(
        self,
        front_photo: str | Path,
        garment_photos: list[str | Path],
        category: str = "tops",
        garment_photo_type: str = "flat-lay",
        side_photo: str | Path | None = None,
        back_photo: str | Path | None = None,
        height_cm: float | None = None,
        weight_kg: float | None = None,
        fabric: str = "cotton",
    ) -> dict:
        """
        Execute the full VTON pipeline.

        Args:
            front_photo: path to front-facing full-body photo (person image for VTON front view)
            garment_photos: list of garment photo paths
            category: garment category ("tops", "bottoms", "one-pieces")
            garment_photo_type: "flat-lay" or "model"
            side_photo: optional side photo for measurements
            back_photo: optional back photo (person image for VTON back view)
            height_cm: user height in cm
            weight_kg: user weight in kg
            fabric: fabric type key

        Returns:
            dict with sizing result, try-on image paths, feed video path, pipeline log
        """
        front_photo = Path(front_photo)

        # Stage 0: Person regeneration
        logger.info("=== Stage 0: Person Regeneration ===")
        person_for_tryon = front_photo  # default fallback
        regenerated_path = self.output_dir / "person_regenerated.png"
        try:
            from server.core.person_regenerator import regenerate_person
            from PIL import Image

            webcam_img = Image.open(front_photo).convert("RGB")
            regen = regenerate_person(webcam_img, regenerated_path)
            if regen:
                person_for_tryon = regenerated_path
                logger.info("Using regenerated person image")
            else:
                logger.info("Regeneration unavailable, falling back to enhancer")
                try:
                    from server.core.person_enhancer import prepare_person_for_vton

                    enhanced = prepare_person_for_vton(webcam_img)
                    enhanced_path = self.output_dir / "person_enhanced.png"
                    enhanced.save(str(enhanced_path))
                    person_for_tryon = enhanced_path
                    logger.info("Using enhanced person image (bg removal + upscale)")
                except Exception as e2:
                    logger.warning("Enhancement also failed (%s), using original photo", e2)
        except Exception as e:
            logger.warning("Regeneration failed (%s), trying enhancer", e)
            try:
                from server.core.person_enhancer import prepare_person_for_vton
                from PIL import Image

                webcam_img = Image.open(front_photo).convert("RGB")
                enhanced = prepare_person_for_vton(webcam_img)
                enhanced_path = self.output_dir / "person_enhanced.png"
                enhanced.save(str(enhanced_path))
                person_for_tryon = enhanced_path
                logger.info("Using enhanced person image (bg removal + upscale)")
            except Exception as e2:
                logger.warning("Enhancement also failed (%s), using original photo", e2)

        # Flush VRAM after Stage 0 before loading FASHN VTON
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Stage 1: Body measurements (uses ORIGINAL front_photo for landmarks)
        logger.info("=== Stage 1: Body Measurements ===")
        measurements = extract_measurements(
            front_photo,
            side_photo=side_photo,
            height_cm=height_cm,
            weight_kg=weight_kg,
            pipeline_log=self.pipeline_log,
        )
        logger.info("Measurements: %s", measurements)

        # Stage 2: Virtual try-on (front + back per garment)
        logger.info("=== Stage 2: Virtual Try-On ===")
        import datetime
        with open("/tmp/vton_debug.log", "a") as _dbg:
            _dbg.write(f"\n[{datetime.datetime.now()}] Stage 2: garment_photos={garment_photos}\n")
            _dbg.write(f"  person_for_tryon={person_for_tryon}\n")
        tryon_results = {}  # {garment_idx: {front: path, back: path}}
        front_tryon_paths = []

        for i, gpath in enumerate(garment_photos):
            gpath = Path(gpath)
            if not gpath.exists():
                logger.warning("Garment photo not found: %s", gpath)
                continue

            garment_results = {}

            # Front view (use regenerated/enhanced person)
            front_out = self.output_dir / f"tryon_front_{i}.png"
            front_result = self.tryon.generate(
                person_path=person_for_tryon,
                garment_path=gpath,
                category=category,
                output_path=front_out,
                garment_photo_type=garment_photo_type,
                pipeline_log=self.pipeline_log,
            )
            if front_result:
                garment_results["front"] = str(front_result)
                front_tryon_paths.append(front_result)

            # Back view
            if back_photo and Path(back_photo).exists():
                back_out = self.output_dir / f"tryon_back_{i}.png"
                back_result = self.tryon.generate(
                    person_path=back_photo,
                    garment_path=gpath,
                    category=category,
                    output_path=back_out,
                    garment_photo_type=garment_photo_type,
                    pipeline_log=self.pipeline_log,
                )
                if back_result:
                    garment_results["back"] = str(back_result)

            tryon_results[i] = garment_results

        # Stage 3: Size recommendation
        logger.info("=== Stage 3: Size Recommendation ===")
        sizing_result = recommend_size(measurements, fabric=fabric)

        # Stage 4: Feed video (front views only)
        logger.info("=== Stage 4: Feed Video ===")
        feed_path = None
        if front_tryon_paths:
            feed_out = self.output_dir / "feed.mp4"
            feed_path = generate_feed_video(
                front_tryon_paths, feed_out,
                pipeline_log=self.pipeline_log,
            )

        # Assemble result
        result = {
            **sizing_result,
            "body_measurements": measurements,
            "tryon_results": tryon_results,
            "feed_video": str(feed_path) if feed_path else None,
            "pipeline_log": self.pipeline_log.to_dicts(),
        }

        # Save result
        result_path = self.output_dir / "sizing_result.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Result saved to %s", result_path)
        logger.info(
            "Recommended size: %s (confidence: %s)",
            result["recommended_size"], result["confidence_score"],
        )

        return result


def main():
    parser = argparse.ArgumentParser(description="Olvon VTON Pipeline")
    parser.add_argument("--front-photo", required=True, help="Path to front-facing full-body photo")
    parser.add_argument("--side-photo", default=None, help="Path to side photo (for measurements)")
    parser.add_argument("--back-photo", default=None, help="Path to back photo (for back view try-on)")
    parser.add_argument("--garment-photo", action="append", default=[], help="Path to garment photo (repeatable)")
    parser.add_argument("--category", default="tops", choices=["tops", "bottoms", "one-pieces"], help="Garment category")
    parser.add_argument("--garment-type", default="flat-lay", choices=["flat-lay", "model"], help="Garment photo type")
    parser.add_argument("--height", type=float, default=None, help="User height in cm")
    parser.add_argument("--weight", type=float, default=None, help="User weight in kg")
    parser.add_argument("--fabric", default="cotton", help="Fabric type")
    parser.add_argument("--output-dir", default="server/outputs", help="Output directory")
    parser.add_argument("--weights-dir", default="server/lib/fashn-vton/weights", help="VTON weights directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    pipeline = VTONPipeline(
        output_dir=args.output_dir,
        weights_dir=args.weights_dir,
    )
    result = pipeline.run(
        front_photo=args.front_photo,
        garment_photos=args.garment_photo,
        category=args.category,
        garment_photo_type=args.garment_type,
        side_photo=args.side_photo,
        back_photo=args.back_photo,
        height_cm=args.height,
        weight_kg=args.weight,
        fabric=args.fabric,
    )

    print("\n=== Pipeline Complete ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
