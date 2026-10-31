"""
PlantHealth AI - Inference Engine
Reusable, robust prediction engine that loads trained Keras models,
validates leaf imagery, performs model-specific preprocessing, and generates
calibrated confidence distributions across all 38 PlantVillage classes.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import tensorflow as tf

from config import (
    CLASS_INDICES_PATH,
    CLASS_METADATA,
    IMAGE_SIZE,
    MOBILENET_MODEL_PATH,
    CUSTOM_CNN_MODEL_PATH,
    VGG16_MODEL_PATH,
    NUM_CLASSES,
)
from src.preprocessing import ImageValidationError, preprocess_for_model, validate_and_load_image
from src.utils import load_json, logger, parse_class_name


class PlantDiseasePredictor:
    """
    Production-grade inference engine for plant leaf disease diagnosis.
    Handles image validation, model-specific preprocessing, class mapping,
    and returns rich top-k diagnosis reports with agricultural prevention guidance.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = MOBILENET_MODEL_PATH,
        class_indices_path: Union[str, Path] = CLASS_INDICES_PATH,
        model_type: str = "mobilenetv2"
    ):
        self.model_path = Path(model_path)
        self.class_indices_path = Path(class_indices_path)
        self.model_type = model_type.lower()
        self.model: Optional[tf.keras.Model] = None
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

        self._load_class_indices()
        self._load_model()

    def _load_class_indices(self) -> None:
        """Loads canonical class-to-index mapping from disk or defaults to CLASS_METADATA."""
        if self.class_indices_path.exists():
            try:
                self.class_to_idx = load_json(self.class_indices_path)
                self.idx_to_class = {int(v): k for k, v in self.class_to_idx.items()}
                logger.info(f"Loaded {len(self.class_to_idx)} class indices from {self.class_indices_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to read {self.class_indices_path}: {e}. Falling back to metadata.")

        # Fallback to sorted metadata keys
        sorted_keys = sorted(CLASS_METADATA.keys())
        self.class_to_idx = {name: idx for idx, name in enumerate(sorted_keys)}
        self.idx_to_class = {idx: name for idx, name in enumerate(sorted_keys)}

    def _load_model(self) -> None:
        """Loads trained Keras model if artifact exists on disk."""
        if not self.model_path.exists():
            logger.warning(f"Model checkpoint not found at {self.model_path}. Inference disabled until trained.")
            self.model = None
            return

        try:
            logger.info(f"Loading trained model from {self.model_path}...")
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Model successfully loaded. Input shape: {self.model.input_shape}, Output: {self.model.output_shape}")
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {e}")
            self.model = None

    @property
    def is_ready(self) -> bool:
        """Returns True if the model and class mapping are successfully loaded."""
        return self.model is not None and len(self.idx_to_class) > 0

    def predict(
        self,
        image_source: Union[str, Path, bytes, Image.Image],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference on a single image.
        
        Args:
            image_source: Filepath, bytes, or PIL Image.
            top_k: Number of highest-confidence predictions to return.
            
        Returns:
            Dictionary containing:
                - is_ready: bool
                - top_prediction: Dict[str, Any]
                - top_k_predictions: List[Dict[str, Any]]
                - raw_probabilities: Dict[str, float]
                - image_preview: PIL.Image
        """
        if not self.is_ready:
            raise RuntimeError(
                f"Model artifact not loaded from '{self.model_path}'. "
                "Please train the model first before running inference."
            )

        # 1. Validation and image resizing
        img_array, pil_img = validate_and_load_image(image_source, target_size=IMAGE_SIZE)

        # 2. Model-specific normalization
        preprocessed_batch = preprocess_for_model(img_array, model_type=self.model_type)

        # 3. Model forward pass
        raw_probs = self.model.predict(preprocessed_batch, verbose=0)[0]
        
        # Ensure probabilities sum to 1.0 (softmax sanity check)
        raw_probs = np.asarray(raw_probs, dtype=np.float32)
        if raw_probs.sum() > 0:
            raw_probs = raw_probs / raw_probs.sum()

        # 4. Extract top-k predictions
        top_indices = np.argsort(raw_probs)[::-1][:top_k]

        top_k_list = []
        for idx in top_indices:
            class_name = self.idx_to_class.get(int(idx), f"Class_{idx}")
            prob = float(raw_probs[idx])
            parsed = parse_class_name(class_name)
            top_k_list.append({
                "class_name": class_name,
                "plant": parsed["plant"],
                "disease": parsed["disease"],
                "status": parsed["status"],
                "confidence": prob,
                "percentage": f"{prob * 100:.2f}%",
                "pathogen": parsed.get("pathogen", "N/A"),
                "description": parsed.get("description", ""),
                "prevention": parsed.get("prevention", [])
            })

        best = top_k_list[0]

        # Full probability dictionary for all 38 classes
        all_probs_dict = {
            self.idx_to_class.get(i, f"Class_{i}"): float(raw_probs[i])
            for i in range(len(raw_probs))
        }

        return {
            "is_ready": True,
            "top_prediction": best,
            "top_k_predictions": top_k_list,
            "all_probabilities": all_probs_dict,
            "image_preview": pil_img
        }


def run_cli_prediction(image_path: str, model_type: str = "mobilenetv2") -> None:
    """CLI helper to run single-image prediction and print results."""
    m_path = MOBILENET_MODEL_PATH if "mobilenet" in model_type else CUSTOM_CNN_MODEL_PATH
    predictor = PlantDiseasePredictor(model_path=m_path, model_type=model_type)

    if not predictor.is_ready:
        print(f"Error: Model not available at {m_path}.")
        return

    result = predictor.predict(image_path)
    best = result["top_prediction"]
    print("=" * 60)
    print("PLANT HEALTH AI - DIAGNOSIS RESULT")
    print("=" * 60)
    print(f"Plant:        {best['plant']}")
    print(f"Condition:    {best['disease']}")
    print(f"Status:       {best['status']}")
    print(f"Confidence:   {best['percentage']}")
    print(f"Pathogen:     {best['pathogen']}")
    print(f"Description:  {best['description']}")
    print("\nPrevention Guidelines:")
    for tip in best["prevention"]:
        print(f" - {tip}")
    print("\nTop 3 Predictions:")
    for i, p in enumerate(result["top_k_predictions"], 1):
        print(f" {i}. {p['plant']} - {p['disease']} ({p['percentage']})")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    m_arg = sys.argv[2] if len(sys.argv) > 2 else "mobilenetv2"
    if img_arg:
        run_cli_prediction(img_arg, m_arg)
    else:
        print("Usage: python -m src.predict <image_path> [mobilenetv2|custom_cnn]")
