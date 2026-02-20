"""
Orchestrator pipeline: body inference -> garment generation -> physics sim -> sizing.
"""

import argparse
import json
import logging
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

from server.core.anny_inference import BodyMeshInference
from server.core.diagnostics import PipelineLog, log_fallback
from server.core.garment_3dgen import Garment3DGenWrapper
from server.core.garment_generator import GarmentGenerator
from server.core.sizing_logic import recommend_size

logger = logging.getLogger(__name__)


class PhysicsTestPipeline:
    """End-to-end pipeline: body mesh -> garment -> physics sim -> sizing."""

    def __init__(
        self,
        input_dir: str | Path = "server/inputs",
        output_dir: str | Path = "server/outputs",
        weights_dir: str | Path = "server/assets/weights",
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.weights_dir = Path(weights_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.input_dir / "garment_photos").mkdir(parents=True, exist_ok=True)

        self.pipeline_log = PipelineLog()

        self.body_inference = BodyMeshInference(self.weights_dir)
        self.body_inference._pipeline_log = self.pipeline_log

        self.garment_generator = GarmentGenerator(self.weights_dir)
        self.garment_generator._pipeline_log = self.pipeline_log

        self.garment_3dgen = Garment3DGenWrapper()
        self.garment_3dgen._pipeline_log = self.pipeline_log

    def _extract_measurements_from_mesh(self, mesh_path: Path) -> dict:
        """
        Approximate body measurements from mesh bounding box.
        chest = x_extent * pi (circumference approximation)
        height = y_extent
        waist = chest * 0.85 (approximate ratio)
        """
        mesh = trimesh.load(str(mesh_path))
        bbox = mesh.bounding_box.extents  # (x, y, z)

        chest_circumference = bbox[0] * math.pi * 100  # convert m -> cm
        height_cm = bbox[1] * 100
        waist_circumference = chest_circumference * 0.85
        hip_circumference = chest_circumference * 1.0

        return {
            "chest": round(chest_circumference, 1),
            "waist": round(waist_circumference, 1),
            "hip": round(hip_circumference, 1),
            "height": round(height_cm, 1),
        }

    def _run_physics_sim(self, body_obj: Path, garment_obj: Path, output_glb: Path, frames: int = 40) -> Path:
        """Run Blender physics simulation as a subprocess."""
        blender_path = shutil.which("blender")
        if not blender_path:
            raise FileNotFoundError(
                "Blender not found on PATH. Install Blender 3.6+ and ensure 'blender' is accessible."
            )

        script_path = Path(__file__).parent / "core" / "physics_sim.py"

        cmd = [
            blender_path,
            "--background",
            "--python", str(script_path),
            "--",
            "--body", str(body_obj.resolve()),
            "--garment", str(garment_obj.resolve()),
            "--output", str(output_glb.resolve()),
            "--frames", str(frames),
        ]

        logger.info("Running Blender: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            logger.error("Blender stdout:\n%s", result.stdout)
            logger.error("Blender stderr:\n%s", result.stderr)
            raise RuntimeError(
                f"Blender physics simulation failed (exit code {result.returncode}): "
                f"{result.stderr[:300]}"
            )

        logger.info("Blender output:\n%s", result.stdout)
        return output_glb

    def run(
        self,
        user_height_cm: float | None = None,
        user_weight_kg: float | None = None,
        garment_photo_path: str | Path | None = None,
        garment_measurements: dict | None = None,
        fabric: str = "cotton",
    ) -> dict:
        """
        Execute the full pipeline. Returns sizing result dict.

        Args:
            user_height_cm: user height in cm (optional but recommended)
            user_weight_kg: user weight in kg (optional)
            garment_photo_path: path to garment photo for Garment3DGen
            garment_measurements: dict with chest_width_cm, body_length_cm, etc.
            fabric: fabric type key
        """
        body_obj = self.output_dir / "temp_body.obj"
        garment_obj = self.output_dir / "temp_garment.obj"
        final_glb = self.output_dir / "final_fitted_avatar.glb"

        # Stage 1: Body mesh inference
        logger.info("=== Stage 1: Body Mesh Inference ===")
        self.body_inference.generate(self.input_dir, body_obj, height_cm=user_height_cm)

        # Stage 2: Garment generation
        logger.info("=== Stage 2: Garment Generation ===")
        garment_generated = False

        # Try Garment3DGen from photo first
        if garment_photo_path is not None:
            photo_path = Path(garment_photo_path)
            if photo_path.exists():
                result_path = self.garment_3dgen.generate_from_photo(
                    photo_path, garment_obj, body_obj
                )
                if result_path is not None:
                    garment_generated = True
                    logger.info("Garment generated via Garment3DGen from photo")

        # Try measurement-based parametric if photo failed or not provided
        if not garment_generated and garment_measurements:
            self.garment_generator.generate_from_measurements(
                garment_obj,
                chest_width_cm=garment_measurements.get("chest_width_cm", 50.0),
                body_length_cm=garment_measurements.get("body_length_cm", 70.0),
                sleeve_length_cm=garment_measurements.get("sleeve_length_cm", 20.0),
                waist_width_cm=garment_measurements.get("waist_width_cm", 48.0),
            )
            garment_generated = True
            logger.info("Garment generated from explicit measurements")

        # Fall back to body-relative parametric
        if not garment_generated:
            self.garment_generator.generate(body_obj, garment_obj)
            logger.info("Garment generated via parametric fallback")

        # Stage 3: Physics simulation (requires Blender)
        logger.info("=== Stage 3: Physics Simulation ===")
        try:
            self._run_physics_sim(body_obj, garment_obj, final_glb)
        except (FileNotFoundError, RuntimeError) as e:
            log_fallback(logger, "physics_sim", e, self.pipeline_log)
            logger.info("To complete this stage, install Blender 3.6+ and add to PATH.")

        # Stage 4: Sizing recommendation
        logger.info("=== Stage 4: Size Recommendation ===")
        measurements = self._extract_measurements_from_mesh(body_obj)

        # Blend in user-provided height/weight
        if user_height_cm is not None:
            measurements["height_cm"] = user_height_cm
        if user_weight_kg is not None:
            measurements["weight_kg"] = user_weight_kg

        sizing_result = recommend_size(measurements, fabric=fabric)
        sizing_result["body_measurements_from_mesh"] = self._extract_measurements_from_mesh(body_obj)
        sizing_result["pipeline_log"] = self.pipeline_log.to_dicts()

        # Save sizing result
        sizing_path = self.output_dir / "sizing_result.json"
        with open(sizing_path, "w") as f:
            json.dump(sizing_result, f, indent=2)
        logger.info("Sizing result saved to %s", sizing_path)
        logger.info("Recommended size: %s (confidence: %s)", sizing_result["recommended_size"], sizing_result["confidence_score"])

        return sizing_result


def main():
    parser = argparse.ArgumentParser(description="Olvon Physics Test Pipeline")
    parser.add_argument("--input-dir", default="server/inputs", help="Directory with captured images")
    parser.add_argument("--output-dir", default="server/outputs", help="Output directory for meshes and results")
    parser.add_argument("--weights-dir", default="server/assets/weights", help="Directory with model weights")
    parser.add_argument("--height", type=float, default=None, help="User height in cm")
    parser.add_argument("--weight", type=float, default=None, help="User weight in kg")
    parser.add_argument("--garment-photo", default=None, help="Path to garment photo for 3D reconstruction")
    parser.add_argument("--fabric", default="cotton", help="Fabric type (cotton, spandex, polyester, etc.)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    pipeline = PhysicsTestPipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        weights_dir=args.weights_dir,
    )
    result = pipeline.run(
        user_height_cm=args.height,
        user_weight_kg=args.weight,
        garment_photo_path=args.garment_photo,
        fabric=args.fabric,
    )

    print("\n=== Pipeline Complete ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
