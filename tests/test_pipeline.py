"""
Tests for the 2D VTON pipeline components.
Tests body measurements, sizing logic, try-on worker fallback, and feed generator.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from server.core.body_measurements import (
    POPULATION_AVERAGES,
    extract,
    _extract_from_landmarks,
)
from server.core.feed_generator import build_ffmpeg_cmd
from server.core.sizing_logic import compute_fit_score, recommend_size
from server.core.tryon_worker import TryOnWorker


class TestBodyMeasurements:
    """Test body measurement extraction with fallback chain."""

    def test_population_average_fallback(self):
        """When no photo and no height, returns population averages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_photo = Path(tmpdir) / "nonexistent.jpg"
            result = extract(fake_photo)

            assert result["chest"] == POPULATION_AVERAGES["chest"]
            assert result["waist"] == POPULATION_AVERAGES["waist"]
            assert result["hip"] == POPULATION_AVERAGES["hip"]

    def test_height_weight_empirical_fallback(self):
        """When photo unavailable but height/weight given, uses empirical formulas."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_photo = Path(tmpdir) / "nonexistent.jpg"
            result = extract(fake_photo, height_cm=175.0, weight_kg=75.0)

            assert "chest" in result
            assert "waist" in result
            assert "hip" in result
            assert result.get("height_cm") == 175.0
            assert result.get("weight_kg") == 75.0
            assert "bmi" in result
            # BMI for 75kg / 1.75m = ~24.5
            assert 20 < result["bmi"] < 30

    def test_height_only_fallback(self):
        """Height-only (no weight) uses empirical with assumed average BMI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_photo = Path(tmpdir) / "nonexistent.jpg"
            result = extract(fake_photo, height_cm=180.0)

            assert "chest" in result
            assert result.get("height_cm") == 180.0
            # No weight → no BMI
            assert "bmi" not in result
            # Should not be population average (empirical should give different values)
            assert result["chest"] != POPULATION_AVERAGES["chest"]

    def test_extract_returns_shoulder_width(self):
        """Result always includes shoulder_width."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_photo = Path(tmpdir) / "nonexistent.jpg"
            result = extract(fake_photo, height_cm=170.0)
            assert "shoulder_width" in result

    def test_extract_from_landmarks_returns_none_without_landmarks(self):
        """_extract_from_landmarks returns None when landmarks are None."""
        assert _extract_from_landmarks(None, 720, 175.0) is None

    def test_extract_from_landmarks_returns_none_for_short_list(self):
        """_extract_from_landmarks returns None for too few landmarks."""
        assert _extract_from_landmarks([], 720, 175.0) is None


class TestSizingLogic:
    """Test the sizing recommendation engine."""

    def test_fit_score_perfect_match(self):
        score = compute_fit_score(100, 100, "cotton")
        assert score == 1.0

    def test_fit_score_poor_match(self):
        score = compute_fit_score(100, 60, "cotton")
        assert score < 0.5

    def test_fit_score_elasticity_helps(self):
        rigid = compute_fit_score(110, 100, "cotton")
        stretchy = compute_fit_score(110, 100, "spandex")
        assert stretchy >= rigid

    def test_recommend_size_returns_correct_structure(self):
        body = {"chest": 99, "waist": 83, "hip": 99}
        result = recommend_size(body)

        assert "recommended_size" in result
        assert "confidence_score" in result
        assert "fit_score" in result
        assert "measurements" in result
        assert "all_scores" in result
        assert result["recommended_size"] in ["S", "M", "L", "XL", "XXL"]

    def test_recommend_size_medium_person(self):
        body = {"chest": 99, "waist": 83, "hip": 99}
        result = recommend_size(body)
        assert result["recommended_size"] == "M"

    def test_recommend_size_serializable(self):
        body = {"chest": 110, "waist": 95, "hip": 110}
        result = recommend_size(body)
        json_str = json.dumps(result)
        assert len(json_str) > 0

    def test_recommend_size_with_height(self):
        body = {"chest": 99, "waist": 83, "hip": 99, "height_cm": 180}
        result = recommend_size(body)

        assert result["recommended_size"] in ["S", "M", "L", "XL", "XXL"]
        assert "height_cm" in result["measurements"]
        assert "bmi" not in result["measurements"]

    def test_recommend_size_with_height_and_weight(self):
        body = {"chest": 99, "waist": 83, "hip": 99, "height_cm": 175, "weight_kg": 75}
        result = recommend_size(body)

        assert result["recommended_size"] in ["S", "M", "L", "XL", "XXL"]
        assert "bmi" in result["measurements"]
        assert "height_cm" in result["measurements"]
        assert "weight_kg" in result["measurements"]
        assert 20 < result["measurements"]["bmi"] < 30

    def test_recommend_size_backward_compatible(self):
        body = {"chest": 99, "waist": 83, "hip": 99}
        result = recommend_size(body)

        assert result["recommended_size"] == "M"
        assert "bmi" not in result["measurements"]
        assert "height_cm" not in result["measurements"]


class TestTryOnWorker:
    """Test TryOnWorker fallback behavior."""

    def test_unavailable_returns_none(self):
        """Worker returns None when FASHN VTON is not installed."""
        worker = TryOnWorker(weights_dir="nonexistent/weights")
        with tempfile.TemporaryDirectory() as tmpdir:
            result = worker.generate(
                person_path=Path(tmpdir) / "person.jpg",
                garment_path=Path(tmpdir) / "garment.jpg",
                category="tops",
                output_path=Path(tmpdir) / "output.png",
            )
            assert result is None

    def test_unavailable_sets_available_false(self):
        """After failed load, _available is False."""
        worker = TryOnWorker(weights_dir="nonexistent/weights")
        worker._try_load_pipeline()
        assert worker._available is False

    def test_generate_logs_fallback(self):
        """Fallback is logged via pipeline_log."""
        from server.core.diagnostics import PipelineLog

        worker = TryOnWorker(weights_dir="nonexistent/weights")
        log = PipelineLog()
        with tempfile.TemporaryDirectory() as tmpdir:
            worker.generate(
                person_path=Path(tmpdir) / "person.jpg",
                garment_path=Path(tmpdir) / "garment.jpg",
                category="tops",
                output_path=Path(tmpdir) / "output.png",
                pipeline_log=log,
            )
        assert len(log.entries) > 0
        assert log.entries[0].stage == "tryon"


class TestFeedGenerator:
    """Test FFmpeg command building for feed video."""

    def test_build_cmd_empty_list(self):
        cmd = build_ffmpeg_cmd([], "output.mp4")
        assert cmd == []

    def test_build_cmd_single_image(self):
        cmd = build_ffmpeg_cmd(["img1.png"], "output.mp4", duration=3.0)
        assert cmd[0] == "ffmpeg"
        assert "-y" in cmd
        assert "img1.png" in " ".join(cmd)
        assert "output.mp4" in " ".join(cmd)

    def test_build_cmd_multiple_images(self):
        cmd = build_ffmpeg_cmd(
            ["img1.png", "img2.png", "img3.png"],
            "output.mp4", duration=3.0, fade=0.5,
        )
        assert cmd[0] == "ffmpeg"
        assert "-filter_complex" in cmd
        assert "xfade" in " ".join(cmd)
        # All inputs present
        cmd_str = " ".join(cmd)
        for img in ["img1.png", "img2.png", "img3.png"]:
            assert img in cmd_str

    def test_build_cmd_two_images(self):
        cmd = build_ffmpeg_cmd(["a.png", "b.png"], "out.mp4")
        assert cmd[0] == "ffmpeg"
        assert "-filter_complex" in cmd

    @patch("shutil.which", return_value=None)
    def test_generate_returns_none_without_ffmpeg(self, mock_which):
        from server.core.feed_generator import generate_feed_video
        result = generate_feed_video(["img.png"], "out.mp4")
        assert result is None
