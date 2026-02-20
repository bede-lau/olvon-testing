"""
Garment mesh generation via ML model or parametric fallback.
Generates a T-shirt mesh sized relative to the body mesh.
"""

import logging
from pathlib import Path

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


class GarmentGenerator:
    """
    Generates a 3D garment mesh (T-shirt).
    Attempts real model weights first; falls back to parametric box mesh.
    """

    def __init__(self, weights_dir: str | Path):
        self.weights_dir = Path(weights_dir)
        self.checkpoint_path = self.weights_dir / "garment_checkpoint.pth"
        self.model = None
        self.device = "cpu"

    def _try_load_model(self) -> bool:
        """Attempt to load the garment model checkpoint."""
        if not self.checkpoint_path.exists():
            logger.info("No garment checkpoint found at %s, will use fallback", self.checkpoint_path)
            return False

        try:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading garment checkpoint on %s...", self.device)
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self.model = checkpoint
            logger.info("Garment model loaded successfully")
            return True
        except Exception as e:
            from server.core.diagnostics import log_fallback
            log_fallback(logger, "garment_model_load", e, getattr(self, '_pipeline_log', None))
            self.model = None
            return False

    def _generate_parametric_tshirt(self, body_mesh_path: str | Path) -> trimesh.Trimesh:
        """
        Generate a parametric T-shirt mesh scaled to the body.
        The garment is positioned above the body for physics drop.
        """
        # Load body mesh to get bounding box
        try:
            body = trimesh.load(str(body_mesh_path))
            bbox = body.bounding_box.extents  # (x, y, z) dimensions
            body_top = body.bounds[1][1]  # max y
            body_center_x = body.centroid[0]
            body_center_z = body.centroid[2]
        except Exception as e:
            logger.warning("Could not load body mesh for sizing: %s. Using defaults.", e)
            bbox = np.array([0.30, 1.70, 0.20])
            body_top = 0.85
            body_center_x = 0.0
            body_center_z = 0.0

        # Scale garment to body + 10% ease
        scale_factor = 1.1
        torso_width = bbox[0] * scale_factor
        torso_height = bbox[1] * 0.4  # T-shirt covers ~40% of body height
        torso_depth = bbox[2] * scale_factor

        # Main body of t-shirt
        shirt_body = trimesh.creation.box(
            extents=[torso_width, torso_height, torso_depth]
        )

        # Sleeves
        sleeve_length = torso_width * 0.4
        sleeve_width = torso_height * 0.25
        sleeve_depth = torso_depth * 0.8

        left_sleeve = trimesh.creation.box(
            extents=[sleeve_length, sleeve_width, sleeve_depth]
        )
        left_sleeve.apply_translation([-(torso_width / 2 + sleeve_length / 2), torso_height * 0.2, 0])

        right_sleeve = trimesh.creation.box(
            extents=[sleeve_length, sleeve_width, sleeve_depth]
        )
        right_sleeve.apply_translation([(torso_width / 2 + sleeve_length / 2), torso_height * 0.2, 0])

        garment = trimesh.util.concatenate([shirt_body, left_sleeve, right_sleeve])

        # Position garment 0.3m above the body top for physics drop
        drop_height = body_top + 0.3
        garment.apply_translation([body_center_x, drop_height + torso_height / 2, body_center_z])

        logger.info(
            "Parametric T-shirt: %.2f x %.2f x %.2f, positioned at y=%.2f",
            torso_width, torso_height, torso_depth, drop_height,
        )
        return garment

    def generate(self, body_mesh_path: str | Path, output_path: str | Path) -> Path:
        """
        Generate a garment mesh.

        Args:
            body_mesh_path: path to the body OBJ mesh
            output_path: where to save the garment OBJ

        Returns:
            Path to the generated garment OBJ file
        """
        output_path = Path(output_path)

        loaded = self._try_load_model()

        if loaded and self.model is not None:
            try:
                import torch
                with torch.no_grad():
                    logger.info("Real garment model loaded but inference not implemented yet")
                    logger.info("Falling back to parametric mesh")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                from server.core.diagnostics import log_fallback
                log_fallback(logger, "garment_inference", e, getattr(self, '_pipeline_log', None))

        mesh = self._generate_parametric_tshirt(body_mesh_path)
        mesh.export(str(output_path))
        logger.info("Garment mesh saved to %s (%d vertices)", output_path, len(mesh.vertices))
        return output_path

    def generate_from_measurements(
        self,
        output_path: str | Path,
        chest_width_cm: float = 50.0,
        body_length_cm: float = 70.0,
        sleeve_length_cm: float = 20.0,
        waist_width_cm: float = 48.0,
    ) -> Path:
        """
        Generate a parametric T-shirt mesh from explicit garment measurements.

        Args:
            output_path: where to save the garment OBJ
            chest_width_cm: half-chest width in cm
            body_length_cm: shoulder to hem in cm
            sleeve_length_cm: sleeve length in cm
            waist_width_cm: half-waist width in cm

        Returns:
            Path to the generated garment OBJ file
        """
        output_path = Path(output_path)

        # Convert cm to meters
        chest_w = chest_width_cm / 100 * 2  # full width
        body_h = body_length_cm / 100
        sleeve_l = sleeve_length_cm / 100
        waist_w = waist_width_cm / 100 * 2
        depth = chest_w * 0.4  # approximate depth

        shirt_body = trimesh.creation.box(extents=[chest_w, body_h, depth])

        sleeve_h = body_h * 0.25
        left_sleeve = trimesh.creation.box(extents=[sleeve_l, sleeve_h, depth * 0.8])
        left_sleeve.apply_translation([-(chest_w / 2 + sleeve_l / 2), body_h * 0.2, 0])

        right_sleeve = trimesh.creation.box(extents=[sleeve_l, sleeve_h, depth * 0.8])
        right_sleeve.apply_translation([(chest_w / 2 + sleeve_l / 2), body_h * 0.2, 0])

        garment = trimesh.util.concatenate([shirt_body, left_sleeve, right_sleeve])

        # Position above origin for physics drop
        garment.apply_translation([0, 1.15 + body_h / 2, 0])

        garment.export(str(output_path))
        logger.info(
            "Measurement-based garment: chest=%.0fcm body=%.0fcm sleeve=%.0fcm → %s (%d verts)",
            chest_width_cm, body_length_cm, sleeve_length_cm, output_path, len(garment.vertices),
        )
        return output_path
