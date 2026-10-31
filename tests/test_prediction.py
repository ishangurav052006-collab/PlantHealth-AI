"""
Tests for Class Parsing, Predictor Output Schema, Confidence Calibration, and Taxonomy Mapping.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from config import CLASS_METADATA, NUM_CLASSES
from src.predict import PlantDiseasePredictor
from src.utils import parse_class_name


def test_parse_class_name_canonical():
    """Verifies that all 38 canonical classes parse into valid plant, disease, and status."""
    for raw_name in CLASS_METADATA.keys():
        parsed = parse_class_name(raw_name)
        assert "plant" in parsed and len(parsed["plant"]) > 0
        assert "disease" in parsed and len(parsed["disease"]) > 0
        assert parsed["status"] in ["Healthy", "Diseased"]
        assert "prevention" in parsed
        assert isinstance(parsed["prevention"], list)


def test_parse_class_name_edge_cases():
    """Verifies parsing of complex names with commas and underscores."""
    p1 = parse_class_name("Pepper,_bell___Bacterial_spot")
    assert p1["plant"] == "Bell Pepper"
    assert p1["disease"] == "Bacterial Spot"
    assert p1["status"] == "Diseased"

    p2 = parse_class_name("Tomato___healthy")
    assert p2["plant"] == "Tomato"
    assert p2["disease"] == "None (Healthy)"
    assert p2["status"] == "Healthy"


def test_predictor_uninitialized_behavior():
    """Verifies predictor raises descriptive error when model file does not exist."""
    fake_path = Path("/non/existent/path/model.keras")
    predictor = PlantDiseasePredictor(model_path=fake_path)
    assert not predictor.is_ready
    with pytest.raises(RuntimeError):
        predictor.predict(np.zeros((224, 224, 3), dtype=np.uint8))


def test_predictor_live_model_inference():
    """Tests live inference on the trained MobileNetV2 model checkpoint."""
    from config import MOBILENET_MODEL_PATH
    if not MOBILENET_MODEL_PATH.exists():
        pytest.skip("Trained model checkpoint not found on disk.")

    predictor = PlantDiseasePredictor(model_path=MOBILENET_MODEL_PATH)
    assert predictor.is_ready

    # Dummy image
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    res = predictor.predict(dummy_img, top_k=3)

    assert res["is_ready"] is True
    assert "top_prediction" in res
    assert "top_k_predictions" in res
    assert len(res["top_k_predictions"]) == 3
    assert len(res["all_probabilities"]) == NUM_CLASSES

    # Confidence Range Validation
    top_pred = res["top_prediction"]
    assert 0.0 <= top_pred["confidence"] <= 1.0
    assert "plant" in top_pred and len(top_pred["plant"]) > 0
    assert "disease" in top_pred and len(top_pred["disease"]) > 0

    # Probabilities sum to 1.0
    total_prob = sum(res["all_probabilities"].values())
    assert np.isclose(total_prob, 1.0, atol=1e-3)
