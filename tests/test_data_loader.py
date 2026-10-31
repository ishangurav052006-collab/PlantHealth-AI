"""
Tests for Dataset Discovery, Split Construction, Anti-Leakage Hashing, and tf.data Loading.
"""

from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import pytest

from src.data_loader import (
    _extract_sample_id,
    build_stratified_dataset_splits,
    discover_dataset,
)


def test_extract_sample_id():
    """Verifies that sample root UUID is accurately extracted from complex PlantVillage filenames."""
    # Standard format with ___
    f1 = "00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335_270deg.JPG"
    assert _extract_sample_id(f1) == "00075aa8-d81a-4184-8541-b692b78d398a"

    # Augmented variation of same leaf
    f2 = "00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335_new30degFlipLR.JPG"
    assert _extract_sample_id(f2) == "00075aa8-d81a-4184-8541-b692b78d398a"

    # Corn common rust without ___
    f4 = "RS_Rust 2335_flipLR.JPG"
    assert _extract_sample_id(f4) == "RS_Rust 2335"

    f5 = "RS_Rust 2335.JPG"
    assert _extract_sample_id(f5) == "RS_Rust 2335"


def test_stratified_split_anti_leakage():
    """
    Simulates a mini dataset with multiple camera angles/augmentations of identical leaves.
    Verifies that all images from the same root leaf are strictly placed in the SAME split.
    """
    classes = ["Apple___Apple_scab", "Tomato___healthy"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    items = []
    # Create 20 unique leaf samples for class 0, each with 3 augmented versions
    for leaf_id in range(20):
        uuid_str = f"leaf_{leaf_id:04d}"
        for aug in ["orig", "rot90", "flip"]:
            p = Path(f"/fake/dir/Apple___Apple_scab/{uuid_str}___label_{aug}.JPG")
            items.append((p, "Apple___Apple_scab"))

    # Create 20 unique leaf samples for class 1, each with 3 augmented versions
    for leaf_id in range(20):
        uuid_str = f"leaf_tom_{leaf_id:04d}"
        for aug in ["orig", "rot90", "flip"]:
            p = Path(f"/fake/dir/Tomato___healthy/{uuid_str}___label_{aug}.JPG")
            items.append((p, "Tomato___healthy"))

    train_s, val_s, test_s = build_stratified_dataset_splits(
        items, class_to_idx, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    # Extract sample IDs in each split
    train_ids = {_extract_sample_id(p.name) for p, _ in train_s}
    val_ids = {_extract_sample_id(p.name) for p, _ in val_s}
    test_ids = {_extract_sample_id(p.name) for p, _ in test_s}

    # CRITICAL LEAKAGE TEST: Intersections must be completely empty!
    assert len(train_ids & val_ids) == 0, "Data leakage detected between Train and Validation!"
    assert len(train_ids & test_ids) == 0, "Data leakage detected between Train and Test!"
    assert len(val_ids & test_ids) == 0, "Data leakage detected between Validation and Test!"


def test_discover_dataset_empty_dir_raises():
    """Verifies that an empty directory raises an informative ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError):
            discover_dataset(tmpdir)
