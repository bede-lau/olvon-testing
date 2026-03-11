"""
Virtual try-on worker wrapping FASHN VTON v1.5.

Produces a 2D photorealistic try-on image from a person photo + garment photo.
Falls back gracefully (returns None) when the pipeline is unavailable.
"""

import logging
from pathlib import Path

from server.core.diagnostics import PipelineLog, log_fallback

logger = logging.getLogger(__name__)


class TryOnWorker:
    """Wraps fashn_vton.TryOnPipeline for 2D virtual try-on inference."""

    def __init__(self, weights_dir: str | Path = "server/lib/fashn-vton/weights"):
        self.weights_dir = Path(weights_dir)
        self._pipeline = None
        self._available: bool | None = None  # lazy

    def _try_load_pipeline(self) -> bool:
        """Attempt to load the FASHN VTON pipeline. Returns True on success."""
        import datetime
        _dbg = open("/tmp/vton_debug.log", "a")
        _dbg.write(f"\n[{datetime.datetime.now()}] _try_load_pipeline called\n")
        if self._available is not None:
            _dbg.write(f"  cached: _available={self._available}\n")
            _dbg.close()
            return self._available
        _dbg.write(f"  weights_dir={self.weights_dir}\n")
        try:
            from fashn_vton import TryOnPipeline

            self._pipeline = TryOnPipeline(str(self.weights_dir))
            self._available = True
            _dbg.write("  SUCCESS: pipeline loaded\n")
            logger.info("FASHN VTON pipeline loaded from %s", self.weights_dir)
        except ImportError as e:
            self._available = False
            self._load_error = str(e)
            _dbg.write(f"  FAIL ImportError: {e}\n")
            logger.error(
                "FASHN VTON import failed (is fashn-vton installed? "
                "Run: pip install -e server/lib/fashn-vton): %s", e,
            )
        except Exception as e:
            self._available = False
            self._load_error = str(e)
            _dbg.write(f"  FAIL {type(e).__name__}: {e}\n")
            logger.error("FASHN VTON pipeline failed to load: %s", e)
        _dbg.close()
        return self._available

    def generate(
        self,
        person_path: str | Path,
        garment_path: str | Path,
        category: str,
        output_path: str | Path,
        garment_photo_type: str = "flat-lay",
        pipeline_log: PipelineLog | None = None,
    ) -> Path | None:
        """
        Run virtual try-on inference.

        Args:
            person_path: path to person photo (full body)
            garment_path: path to garment photo
            category: garment category ("tops", "bottoms", "one-pieces")
            output_path: where to save the result image
            garment_photo_type: "flat-lay" or "model"
            pipeline_log: optional structured log collector

        Returns:
            Path to saved try-on image, or None if pipeline unavailable.
        """
        person_path = Path(person_path)
        garment_path = Path(garment_path)
        output_path = Path(output_path)

        import datetime
        with open("/tmp/vton_debug.log", "a") as _dbg:
            _dbg.write(f"\n[{datetime.datetime.now()}] generate() called\n")
            _dbg.write(f"  person={person_path}, garment={garment_path}\n")

        if not self._try_load_pipeline():
            reason = getattr(self, "_load_error", "unknown")
            log_fallback(
                logger, "tryon",
                RuntimeError(f"FASHN VTON pipeline not available: {reason}"),
                pipeline_log,
            )
            return None

        try:
            from PIL import Image

            person_img = Image.open(person_path).convert("RGB")
            garment_img = Image.open(garment_path).convert("RGB")

            result = self._pipeline(
                person_image=person_img,
                garment_image=garment_img,
                category=category,
                garment_photo_type=garment_photo_type,
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.images[0].save(str(output_path))
            logger.info("Try-on result saved to %s", output_path)
            return output_path

        except Exception as e:
            log_fallback(logger, "tryon", e, pipeline_log)
            return None
