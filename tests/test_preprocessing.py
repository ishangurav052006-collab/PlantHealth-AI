"""
Tests for Image Loading, Validation, and Model-Specific Preprocessing.
"""

import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.preprocessing import ImageValidationError, preprocess_for_model, validate_and_load_image


@pytest.fixture
def sample_pil_image():
    """Generates a synthetic 300x400 RGB test image."""
    arr = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_validate_and_load_image_from_pil(sample_pil_image):
    """Verifies resizing and shape conversion from a PIL Image."""
    img_array, pil_img = validate_and_load_image(sample_pil_image, target_size=(224, 224))
    assert isinstance(img_array, np.ndarray)
    assert img_array.shape == (224, 224, 3)
    assert pil_img.size == (224, 224)
    assert img_array.dtype == np.uint8


def test_validate_and_load_image_from_bytes(sample_pil_image):
    """Verifies loading image from in-memory byte buffer."""
    buf = io.BytesIO()
    sample_pil_image.save(buf, format="JPEG")
    raw_bytes = buf.getvalue()

    img_array, pil_img = validate_and_load_image(raw_bytes, target_size=(224, 224))
    assert img_array.shape == (224, 224, 3)
    assert pil_img.mode == "RGB"


def test_validate_and_load_grayscale_conversion():
    """Verifies single-channel grayscale image is converted to 3-channel RGB."""
    gray_arr = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    gray_pil = Image.fromarray(gray_arr, mode="L")

    img_array, pil_img = validate_and_load_image(gray_pil, target_size=(224, 224))
    assert img_array.shape == (224, 224, 3)
    assert pil_img.mode == "RGB"


def test_corrupted_empty_image_raises_error():
    """Verifies empty bytes raise ImageValidationError."""
    with pytest.raises(ImageValidationError):
        validate_and_load_image(b"")


def test_corrupted_invalid_data_raises_error():
    """Verifies non-image data raises ImageValidationError."""
    with pytest.raises(ImageValidationError):
        validate_and_load_image(b"NOT_A_VALID_IMAGE_HEADER_OR_DATA")


def test_custom_cnn_preprocessing_range():
    """Verifies custom CNN normalization scales to [0.0, 1.0]."""
    raw = np.array([[[0, 128, 255]]], dtype=np.uint8)
    processed = preprocess_for_model(raw, model_type="custom_cnn")
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0
    assert processed.shape == (1, 1, 1, 3)


def test_mobilenetv2_preprocessing_range():
    """Verifies MobileNetV2 normalization scales roughly to [-1.0, 1.0]."""
    raw = np.full((224, 224, 3), 255, dtype=np.uint8)
    processed = preprocess_for_model(raw, model_type="mobilenetv2")
    assert processed.shape == (1, 224, 224, 3)
    # 255 should map to ~1.0, 0 should map to ~ -1.0
    assert np.isclose(processed.max(), 1.0, atol=0.05)

    raw_zero = np.zeros((224, 224, 3), dtype=np.uint8)
    processed_zero = preprocess_for_model(raw_zero, model_type="mobilenetv2")
    assert np.isclose(processed_zero.min(), -1.0, atol=0.05)
