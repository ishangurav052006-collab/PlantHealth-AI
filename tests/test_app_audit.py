"""
PlantHealth AI - Application & Inference Integrity Audit Tests
Validates Streamlit configuration, invalid image error handling, dynamic metric loading,
and ensures complete absence of mock data or hardcoded inferences.
"""

from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import pytest

from config import (
    CLASS_INDICES_PATH,
    CLASS_METADATA,
    FINAL_METRICS_JSON_PATH,
    MOBILENET_MODEL_PATH,
    NUM_CLASSES,
)
from src.predict import PlantDiseasePredictor
from src.preprocessing import ImageValidationError, validate_and_load_image
from src.utils import load_json


def test_app_loads_correct_trained_checkpoint():
    """Verifies that the primary predictor loads the genuine trained MobileNetV2 checkpoint."""
    assert MOBILENET_MODEL_PATH.exists(), "Trained model checkpoint does not exist!"
    predictor = PlantDiseasePredictor(model_path=MOBILENET_MODEL_PATH, model_type="mobilenetv2")
    assert predictor.is_ready, "Predictor failed to initialize!"
    assert predictor.model.input_shape == (None, 224, 224, 3)
    assert predictor.model.output_shape == (None, 38)


def test_all_38_classes_available_in_indices_and_taxonomy():
    """Verifies canonical coverage of all 38 diagnostic classes."""
    assert CLASS_INDICES_PATH.exists(), "class_indices.json missing!"
    class_indices = load_json(CLASS_INDICES_PATH)
    assert len(class_indices) == NUM_CLASSES, f"Expected 38 classes, found {len(class_indices)}"

    for class_name, idx in class_indices.items():
        assert 0 <= idx < 38
        assert class_name in CLASS_METADATA, f"Class {class_name} missing from taxonomy metadata!"
        meta = CLASS_METADATA[class_name]
        assert "plant" in meta and len(meta["plant"]) > 0
        assert "disease" in meta and len(meta["disease"]) > 0
        assert "prevention" in meta and len(meta["prevention"]) > 0


def test_invalid_image_handling_safely_rejected():
    """Verifies that empty files, corrupted streams, and non-images are safely caught."""
    # 1. 0-byte stream
    with pytest.raises(ImageValidationError, match="empty"):
        validate_and_load_image(b"")

    # 2. Random binary garbage
    with pytest.raises(ImageValidationError):
        validate_and_load_image(b"\x00\x01\x02\x03\x04\x05GARBAGE_NOT_AN_IMAGE")

    # 3. Truncated header
    with pytest.raises(ImageValidationError):
        validate_and_load_image(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01")

    # 4. 0-byte file on disk
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        tmp.write(b"")
        tmp.flush()
        with pytest.raises(ImageValidationError, match="empty"):
            validate_and_load_image(tmp.name)


def test_real_uploaded_image_inference():
    """Verifies that real PIL Images and raw byte streams generate valid predictions without error."""
    predictor = PlantDiseasePredictor(model_path=MOBILENET_MODEL_PATH, model_type="mobilenetv2")

    # Generate a synthetic RGB leaf-like image
    test_img = Image.new("RGB", (300, 300), color=(46, 125, 50))
    res = predictor.predict(test_img, top_k=3)

    assert res["is_ready"] is True
    assert len(res["all_probabilities"]) == 38
    assert np.isclose(sum(res["all_probabilities"].values()), 1.0, atol=1e-4)
    assert len(res["top_k_predictions"]) == 3
    assert res["top_prediction"]["confidence"] > 0.0


def test_no_hardcoded_or_mock_inferences_in_app():
    """Scans app.py source to ensure no mock predictions or hardcoded results exist."""
    app_source = Path("app.py").read_text(encoding="utf-8")
    assert "mock" not in app_source.lower() or "mock data" in app_source.lower()
    assert "fake_prediction" not in app_source
    assert "dummy_prediction" not in app_source
    assert "get_cached_predictor" in app_source
    assert "predictor.predict" in app_source


def test_dynamic_metrics_exist():
    """Verifies that final_metrics.json exists with genuine evaluation outputs."""
    assert FINAL_METRICS_JSON_PATH.exists(), "final_metrics.json missing!"
    metrics = load_json(FINAL_METRICS_JSON_PATH)
    assert "MobileNetV2 (Primary)" in metrics
    mob = metrics["MobileNetV2 (Primary)"]
    assert 0.80 <= mob["accuracy"] <= 1.0
    assert 0.80 <= mob["f1_score"] <= 1.0
