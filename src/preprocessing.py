"""
PlantHealth AI - Image Preprocessing Pipeline
Provides robust image loading, format validation, corrupted image detection,
resizing to 224x224, and model-specific normalization.
"""

import io
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess

from config import IMAGE_SIZE


class ImageValidationError(Exception):
    """Raised when an uploaded image fails validation or is corrupted."""
    pass


def validate_and_load_image(
    image_source: Union[str, Path, bytes, io.BytesIO, Image.Image],
    target_size: Tuple[int, int] = IMAGE_SIZE
) -> Tuple[np.ndarray, Image.Image]:
    """
    Validates, opens, cleans, and resizes an image into a standard RGB numpy array and PIL Image.
    
    Supports:
        - File path string or Path object
        - Raw bytes or BytesIO stream (e.g. from Streamlit file uploader)
        - Pre-existing PIL Image
        
    Returns:
        Tuple[np.ndarray, Image.Image]:
            - Resized RGB numpy array of shape (H, W, 3) in uint8 [0, 255]
            - Resized PIL Image
            
    Raises:
        ImageValidationError: If image cannot be read, is empty, or corrupted.
    """
    try:
        if isinstance(image_source, np.ndarray):
            pil_img = Image.fromarray(image_source)
        elif isinstance(image_source, Image.Image):
            pil_img = image_source
        elif isinstance(image_source, (str, Path)):
            path = Path(image_source)
            if not path.is_file():
                raise ImageValidationError(f"File does not exist: {path}")
            if path.stat().st_size == 0:
                raise ImageValidationError("Image file is empty (0 bytes).")
            pil_img = Image.open(path)
        elif isinstance(image_source, bytes):
            if len(image_source) == 0:
                raise ImageValidationError("Uploaded byte stream is empty.")
            pil_img = Image.open(io.BytesIO(image_source))
        elif isinstance(image_source, io.BytesIO):
            image_source.seek(0)
            pil_img = Image.open(image_source)
        else:
            raise ImageValidationError(f"Unsupported image input type: {type(image_source)}")

        # Verify integrity for file-based / stream images
        if not isinstance(image_source, (np.ndarray, Image.Image)):
            pil_img.verify()

            # Re-open after verify() because verify() invalidates the image handle in PIL
            if isinstance(image_source, (str, Path)):
                pil_img = Image.open(str(image_source))
            elif isinstance(image_source, bytes):
                pil_img = Image.open(io.BytesIO(image_source))
            elif isinstance(image_source, io.BytesIO):
                image_source.seek(0)
                pil_img = Image.open(image_source)
        elif isinstance(image_source, Image.Image):
            pil_img = image_source.copy()
        elif isinstance(image_source, np.ndarray):
            pil_img = Image.fromarray(image_source)

        # Handle EXIF orientation if present
        pil_img = ImageOps.exif_transpose(pil_img)

        # Convert to standard RGB (drops Alpha/CMYK/Grayscale channels)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # Resize using high-quality Lanczos resampling
        if pil_img.size != target_size:
            pil_img_resized = pil_img.resize(target_size, Image.Resampling.LANCZOS)
        else:
            pil_img_resized = pil_img

        img_array = np.array(pil_img_resized, dtype=np.uint8)

        if img_array.shape != (*target_size, 3):
            raise ImageValidationError(
                f"Unexpected resized shape: {img_array.shape}, expected (*{target_size}, 3)"
            )

        return img_array, pil_img_resized

    except Exception as e:
        if isinstance(e, ImageValidationError):
            raise
        raise ImageValidationError(f"Corrupted or invalid image: {str(e)}") from e


def preprocess_for_model(
    img_array: np.ndarray,
    model_type: str = "mobilenetv2"
) -> np.ndarray:
    """
    Applies the model-specific preprocessing function to an RGB uint8 image array.
    
    Args:
        img_array: np.ndarray of shape (H, W, 3) or (B, H, W, 3) with values in [0, 255]
        model_type: 'mobilenetv2', 'custom_cnn', or 'vgg16'
        
    Returns:
        np.ndarray with batch dimension (1, H, W, 3) or (B, H, W, 3) ready for model.predict()
    """
    model_type = model_type.lower().replace("-", "").replace("_", "")

    # Ensure float32 copy and batch dimension
    x = img_array.astype(np.float32)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)

    if "mobilenet" in model_type:
        # MobileNetV2 normalizes from [0, 255] to [-1, 1]
        return mobilenet_preprocess(x)

    elif "vgg" in model_type:
        # VGG16 converts RGB to BGR and subtracts ImageNet zero-center means
        return vgg16_preprocess(x)

    elif "custom" in model_type:
        # Custom CNN normalizes from [0, 255] to [0.0, 1.0]
        return x / 255.0

    else:
        # Default fallback to standard [0, 1] normalization
        return x / 255.0
