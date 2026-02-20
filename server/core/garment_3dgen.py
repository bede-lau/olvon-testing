"""
Garment3DGen + InstantMesh wrapper for photo-to-3D garment generation.
Follows Real-First-With-Fallback: tries ML pipeline, falls back to parametric.
"""

import logging
import time
from pathlib import Path

import numpy as np
import trimesh

from server.core.diagnostics import PipelineLog, log_fallback

logger = logging.getLogger(__name__)


class Garment3DGenWrapper:
    """
    Wraps Garment3DGen + InstantMesh for photo → 3D garment generation.
    Falls back to parametric generation if dependencies are unavailable.
    """

    STAGES = [
        "dependency_check",
        "instantmesh_recon",
        "garment3dgen_deform",
        "texture_gen",
    ]

    def __init__(
        self,
        garment3dgen_path: str | Path = "server/lib/Garment3DGen",
        instantmesh_path: str | Path = "server/lib/InstantMesh",
    ):
        self.garment3dgen_path = Path(garment3dgen_path)
        self.instantmesh_path = Path(instantmesh_path)
        self.available = False
        self.fallback_log: list[dict] = []
        self._pipeline_log: PipelineLog | None = None
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if Garment3DGen and InstantMesh are available."""
        t0 = time.time()
        try:
            if not self.garment3dgen_path.exists():
                raise FileNotFoundError(
                    f"Garment3DGen not found at {self.garment3dgen_path}. "
                    "Run: git submodule update --init --recursive"
                )
            if not self.instantmesh_path.exists():
                raise FileNotFoundError(
                    f"InstantMesh not found at {self.instantmesh_path}. "
                    "Run: git submodule update --init --recursive"
                )

            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available — Garment3DGen requires GPU")

            self.available = True
            logger.info("Garment3DGen dependencies available (GPU: %s)", torch.cuda.get_device_name(0))
        except Exception as e:
            self.available = False
            log_fallback(logger, "dependency_check", e, self._pipeline_log, time.time() - t0)
            self.fallback_log.append({
                "stage": "dependency_check",
                "error_type": type(e).__name__,
                "message": str(e),
                "gpu_info": "",
            })

    def _run_instantmesh(self, photo_path: Path, output_obj: Path) -> Path:
        """Run InstantMesh: photo → 3D mesh (.obj)."""
        t0 = time.time()
        try:
            import sys
            import subprocess

            script = self.instantmesh_path / "run.py"
            if not script.exists():
                raise FileNotFoundError(f"InstantMesh run.py not found at {script}")

            cmd = [
                sys.executable, str(script),
                str(photo_path),
                "--output_path", str(output_obj.parent),
            ]
            logger.info("Running InstantMesh: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                raise RuntimeError(
                    f"InstantMesh failed (exit {result.returncode}): {result.stderr[:500]}"
                )

            if not output_obj.exists():
                # InstantMesh may output with different naming
                candidates = list(output_obj.parent.glob("*.obj"))
                if candidates:
                    candidates[0].rename(output_obj)
                else:
                    raise FileNotFoundError("InstantMesh produced no .obj output")

            logger.info("InstantMesh reconstruction complete: %s", output_obj)
            return output_obj

        except Exception as e:
            duration = time.time() - t0
            log_fallback(logger, "instantmesh_recon", e, self._pipeline_log, duration)
            self.fallback_log.append({
                "stage": "instantmesh_recon",
                "error_type": type(e).__name__,
                "message": str(e),
                "gpu_info": "",
            })
            raise

    def _run_garment3dgen(self, source_mesh: Path, body_mesh: Path, output_obj: Path) -> Path:
        """Run Garment3DGen: deform source mesh to match body + generate textures."""
        t0 = time.time()
        try:
            import sys
            import subprocess

            script = self.garment3dgen_path / "run_garment3dgen.py"
            if not script.exists():
                # Try alternative entry point
                script = self.garment3dgen_path / "main.py"
            if not script.exists():
                raise FileNotFoundError(
                    f"Garment3DGen entry script not found in {self.garment3dgen_path}"
                )

            cmd = [
                sys.executable, str(script),
                "--source_mesh", str(source_mesh),
                "--target_body", str(body_mesh),
                "--output", str(output_obj),
            ]
            logger.info("Running Garment3DGen: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                raise RuntimeError(
                    f"Garment3DGen failed (exit {result.returncode}): {result.stderr[:500]}"
                )

            if not output_obj.exists():
                raise FileNotFoundError("Garment3DGen produced no output")

            logger.info("Garment3DGen deformation complete: %s", output_obj)
            return output_obj

        except Exception as e:
            duration = time.time() - t0
            log_fallback(logger, "garment3dgen_deform", e, self._pipeline_log, duration)
            self.fallback_log.append({
                "stage": "garment3dgen_deform",
                "error_type": type(e).__name__,
                "message": str(e),
                "gpu_info": "",
            })
            raise

    def generate_from_photo(
        self,
        photo_path: str | Path,
        output_obj_path: str | Path,
        body_mesh_path: str | Path,
    ) -> Path | None:
        """
        Generate a 3D garment from a photo using Garment3DGen + InstantMesh.

        Returns:
            Path to the output OBJ, or None if the pipeline failed (check fallback_log).
        """
        photo_path = Path(photo_path)
        output_obj_path = Path(output_obj_path)
        body_mesh_path = Path(body_mesh_path)
        self.fallback_log = []

        if not self.available:
            logger.info("Garment3DGen not available, skipping photo-based generation")
            return None

        try:
            # Stage 1: InstantMesh — photo → 3D mesh
            intermediate_obj = output_obj_path.parent / "instantmesh_raw.obj"
            self._run_instantmesh(photo_path, intermediate_obj)

            # Stage 2: Garment3DGen — deform mesh to body
            self._run_garment3dgen(intermediate_obj, body_mesh_path, output_obj_path)

            # Position garment 0.3m above body for physics drop
            try:
                body = trimesh.load(str(body_mesh_path))
                garment = trimesh.load(str(output_obj_path))
                body_top = body.bounds[1][1]
                garment_bottom = garment.bounds[0][1]
                offset = (body_top + 0.3) - garment_bottom
                garment.apply_translation([0, offset, 0])
                garment.export(str(output_obj_path))
            except Exception as e:
                logger.warning("Could not reposition garment: %s", e)

            logger.info("Garment3DGen pipeline complete: %s", output_obj_path)
            return output_obj_path

        except Exception:
            # Fallback log already populated by individual stages
            logger.info("Garment3DGen pipeline failed, falling back to parametric generation")
            return None
