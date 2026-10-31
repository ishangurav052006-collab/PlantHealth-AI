"""
Tests for Streamlit Application Components and Navigation.
"""

from unittest.mock import patch
import pytest

from app import CUSTOM_CSS, get_cached_predictor


def test_custom_css_structure():
    """Verifies that the AgTech custom CSS contains the required branding styles."""
    assert "hero-banner" in CUSTOM_CSS
    assert "metric-card" in CUSTOM_CSS
    assert "badge-healthy" in CUSTOM_CSS
    assert "badge-diseased" in CUSTOM_CSS


def test_get_cached_predictor_instantiation():
    """Verifies get_cached_predictor returns a PlantDiseasePredictor instance."""
    predictor = get_cached_predictor("MobileNetV2 (Primary)")
    assert predictor is not None
    assert hasattr(predictor, "predict")
    assert hasattr(predictor, "is_ready")
