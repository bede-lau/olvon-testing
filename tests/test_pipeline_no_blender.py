"""
Smoke tests for the pipeline components that don't require Blender.
Tests body mesh fallback, garment fallback, sizing logic, and Garment3DGen wrapper.
"""

import json
import tempfile
from pathlib import Path

import trimesh

from server.core.anny_inference import BodyMeshInference
from server.core.garment_generator import GarmentGenerator
from server.core.garment_3dgen import Garment3DGenWrapper
from server.core.sizing_logic import compute_fit_score, recommend_size


class TestBodyMeshInference:
    """Test that BodyMeshInference fallback produces a valid mesh."""

    def test_fallback_produces_obj(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_dir = Path(tmpdir) / "weights"
            weights_dir.mkdir()

            inference = BodyMeshInference(weights_dir)
            output_path = Path(tmpdir) / "body.obj"
            result = inference.generate(tmpdir, output_path)

            assert result.exists(), "Output OBJ file was not created"
            mesh = trimesh.load(str(result))
            assert len(mesh.vertices) > 0, "Mesh has no vertices"
            assert len(mesh.faces) > 0, "Mesh has no faces"

    def test_fallback_mesh_has_reasonable_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_dir = Path(tmpdir) / "weights"
            weights_dir.mkdir()

            inference = BodyMeshInference(weights_dir)
            output_path = Path(tmpdir) / "body.obj"
            inference.generate(tmpdir, output_path)

            mesh = trimesh.load(str(output_path))
            extents = mesh.bounding_box.extents
            # Body should be roughly human-proportioned
            assert extents[1] > 0.5, f"Body too short: {extents[1]}"
            assert extents[1] < 3.0, f"Body too tall: {extents[1]}"


class TestGarmentGenerator:
    """Test that GarmentGenerator fallback produces a valid mesh."""

    def test_fallback_produces_obj(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_dir = Path(tmpdir) / "weights"
            weights_dir.mkdir()

            # First generate a body mesh to size against
            body_inference = BodyMeshInference(weights_dir)
            body_path = Path(tmpdir) / "body.obj"
            body_inference.generate(tmpdir, body_path)

            # Generate garment
            generator = GarmentGenerator(weights_dir)
            garment_path = Path(tmpdir) / "garment.obj"
            result = generator.generate(body_path, garment_path)

            assert result.exists(), "Garment OBJ file was not created"
            mesh = trimesh.load(str(result))
            assert len(mesh.vertices) > 0, "Garment mesh has no vertices"
            assert len(mesh.faces) > 0, "Garment mesh has no faces"

    def test_garment_positioned_above_body(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_dir = Path(tmpdir) / "weights"
            weights_dir.mkdir()

            body_inference = BodyMeshInference(weights_dir)
            body_path = Path(tmpdir) / "body.obj"
            body_inference.generate(tmpdir, body_path)

            body_mesh = trimesh.load(str(body_path))
            body_top = body_mesh.bounds[1][1]

            generator = GarmentGenerator(weights_dir)
            garment_path = Path(tmpdir) / "garment.obj"
            generator.generate(body_path, garment_path)

            garment_mesh = trimesh.load(str(garment_path))
            garment_bottom = garment_mesh.bounds[0][1]

            assert garment_bottom > body_top, (
                f"Garment bottom ({garment_bottom:.3f}) should be above body top ({body_top:.3f})"
            )

    def test_generates_garment_from_measurements(self):
        """Parametric method with explicit dimensions produces a valid mesh."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_dir = Path(tmpdir) / "weights"
            weights_dir.mkdir()

            generator = GarmentGenerator(weights_dir)
            garment_path = Path(tmpdir) / "garment_measured.obj"
            result = generator.generate_from_measurements(
                garment_path,
                chest_width_cm=52.0,
                body_length_cm=72.0,
                sleeve_length_cm=22.0,
                waist_width_cm=46.0,
            )

            assert result.exists(), "Measurement-based garment OBJ not created"
            mesh = trimesh.load(str(result))
            assert len(mesh.vertices) > 0, "Garment mesh has no vertices"
            assert len(mesh.faces) > 0, "Garment mesh has no faces"


class TestSizingLogic:
    """Test the sizing recommendation engine."""

    def test_fit_score_perfect_match(self):
        score = compute_fit_score(100, 100, "cotton")
        assert score == 1.0

    def test_fit_score_poor_match(self):
        score = compute_fit_score(100, 60, "cotton")
        assert score < 0.5

    def test_fit_score_elasticity_helps(self):
        # When garment is notably smaller than user, elasticity helps
        rigid = compute_fit_score(110, 100, "cotton")
        stretchy = compute_fit_score(110, 100, "spandex")
        # Spandex stretches from 100 -> 120, closer to 110 than cotton's 100
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
        # Should be JSON-serializable
        json_str = json.dumps(result)
        assert len(json_str) > 0

    def test_recommend_size_with_height(self):
        """Height-only input works and blends with mesh measurements."""
        body = {"chest": 99, "waist": 83, "hip": 99, "height_cm": 180}
        result = recommend_size(body)

        assert result["recommended_size"] in ["S", "M", "L", "XL", "XXL"]
        assert "height_cm" in result["measurements"]
        # BMI should NOT be present when only height given
        assert "bmi" not in result["measurements"]

    def test_recommend_size_with_height_and_weight(self):
        """Height + weight produces BMI in the result."""
        body = {"chest": 99, "waist": 83, "hip": 99, "height_cm": 175, "weight_kg": 75}
        result = recommend_size(body)

        assert result["recommended_size"] in ["S", "M", "L", "XL", "XXL"]
        assert "bmi" in result["measurements"]
        assert "height_cm" in result["measurements"]
        assert "weight_kg" in result["measurements"]
        # BMI for 75kg/1.75m should be ~24.5
        assert 20 < result["measurements"]["bmi"] < 30

    def test_recommend_size_backward_compatible(self):
        """No height/weight still works as before."""
        body = {"chest": 99, "waist": 83, "hip": 99}
        result = recommend_size(body)

        assert result["recommended_size"] == "M"
        assert "bmi" not in result["measurements"]
        assert "height_cm" not in result["measurements"]


class TestGarment3DGenWrapper:
    """Test Garment3DGen wrapper fallback behavior."""

    def test_garment3dgen_wrapper_fallback(self):
        """Wrapper falls back gracefully when dependencies are missing."""
        wrapper = Garment3DGenWrapper(
            garment3dgen_path="nonexistent/Garment3DGen",
            instantmesh_path="nonexistent/InstantMesh",
        )
        assert wrapper.available is False
        assert len(wrapper.fallback_log) > 0
        assert wrapper.fallback_log[0]["stage"] == "dependency_check"

    def test_garment3dgen_generate_returns_none_when_unavailable(self):
        """generate_from_photo returns None when deps are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wrapper = Garment3DGenWrapper(
                garment3dgen_path="nonexistent/Garment3DGen",
                instantmesh_path="nonexistent/InstantMesh",
            )
            result = wrapper.generate_from_photo(
                Path(tmpdir) / "photo.jpg",
                Path(tmpdir) / "garment.obj",
                Path(tmpdir) / "body.obj",
            )
            assert result is None
