"""
Body mesh generation via ANNY parametric model.
Real-first with trimesh fallback when the anny package is not installed.
"""

import logging
from pathlib import Path

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


class BodyMeshInference:
    """
    Generates a 3D body mesh using the ANNY parametric body model.
    Falls back to trimesh primitives when the anny package is unavailable.
    """

    def __init__(self, weights_dir: str | Path = ""):
        self.weights_dir = Path(weights_dir) if weights_dir else Path(".")
        self.model = None
        self.anny_available = False
        self._try_load_anny()

    def _try_load_anny(self) -> bool:
        """Attempt to import anny and create the fullbody model."""
        try:
            import anny as _anny
            import torch

            self.model = _anny.create_fullbody_model(triangulate_faces=True)
            self.anny_available = True
            logger.info("ANNY parametric model loaded successfully")
            return True
        except ImportError:
            logger.info("anny package not installed, will use fallback mesh")
            return False
        except Exception as e:
            from server.core.diagnostics import log_fallback
            log_fallback(logger, "anny_model_load", e, getattr(self, '_pipeline_log', None))
            return False

    def _generate_anny_mesh(self, output_path: Path, height_cm: float | None = None) -> trimesh.Trimesh:
        """Generate a body mesh using the ANNY parametric model."""
        import anny as _anny
        import torch
        from anny.anthropometry import Anthropometry

        phenotype_kwargs = {
            'gender': torch.tensor([0.5]),
            'age': torch.tensor([0.67]),
            'muscle': torch.tensor([0.5]),
            'weight': torch.tensor([0.5]),
            'height': torch.tensor([0.5]),
            'proportions': torch.tensor([0.5]),
        }

        if height_cm is not None:
            anthro = Anthropometry(self.model)
            target_m = height_cm / 100.0

            lo, hi = 0.0, 1.0
            for _ in range(20):
                mid = (lo + hi) / 2.0
                phenotype_kwargs['height'] = torch.tensor([mid])
                output = self.model(phenotype_kwargs=phenotype_kwargs)
                verts = output['vertices'][0].detach().cpu().numpy()
                current_height = verts[:, 1].max() - verts[:, 1].min()
                if current_height < target_m:
                    lo = mid
                else:
                    hi = mid

            phenotype_kwargs['height'] = torch.tensor([(lo + hi) / 2.0])

        output = self.model(phenotype_kwargs=phenotype_kwargs)
        vertices = output['vertices'][0].detach().cpu().numpy()
        faces = self.model.faces.detach().cpu().numpy()

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(str(output_path))
        logger.info("ANNY body mesh saved to %s (%d vertices)", output_path, len(mesh.vertices))
        return mesh

    def _generate_fallback_mesh(self) -> trimesh.Trimesh:
        """Generate a simple humanoid body mesh using trimesh primitives."""
        logger.info("Generating fallback body mesh (cylinder + sphere)")

        # Torso: cylinder
        torso = trimesh.creation.cylinder(radius=0.15, height=0.7, sections=32)
        torso.apply_translation([0, 0.35, 0])

        # Head: sphere
        head = trimesh.creation.icosphere(subdivisions=3, radius=0.10)
        head.apply_translation([0, 0.80, 0])

        # Legs: two cylinders
        left_leg = trimesh.creation.cylinder(radius=0.06, height=0.8, sections=16)
        left_leg.apply_translation([-0.08, -0.40, 0])

        right_leg = trimesh.creation.cylinder(radius=0.06, height=0.8, sections=16)
        right_leg.apply_translation([0.08, -0.40, 0])

        # Arms: two cylinders
        left_arm = trimesh.creation.cylinder(radius=0.04, height=0.6, sections=16)
        left_arm.apply_translation([-0.22, 0.40, 0])

        right_arm = trimesh.creation.cylinder(radius=0.04, height=0.6, sections=16)
        right_arm.apply_translation([0.22, 0.40, 0])

        body = trimesh.util.concatenate([torso, head, left_leg, right_leg, left_arm, right_arm])
        return body

    def generate(self, input_dir: str | Path, output_path: str | Path, height_cm: float | None = None) -> Path:
        """
        Generate a body mesh from input images.

        Args:
            input_dir: directory containing captured images
            output_path: where to save the output OBJ file
            height_cm: optional target height in cm for ANNY model

        Returns:
            Path to the generated OBJ file
        """
        output_path = Path(output_path)

        if self.anny_available and self.model is not None:
            try:
                self._generate_anny_mesh(output_path, height_cm=height_cm)
                return output_path
            except Exception as e:
                from server.core.diagnostics import log_fallback
                log_fallback(logger, "anny_inference", e, getattr(self, '_pipeline_log', None))
                logger.info("Falling back to parametric mesh")

        mesh = self._generate_fallback_mesh()
        mesh.export(str(output_path))
        logger.info("Body mesh saved to %s (%d vertices)", output_path, len(mesh.vertices))
        return output_path
