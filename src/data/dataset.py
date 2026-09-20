import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import logging

logger = logging.getLogger("synthmed.data")

def encode_metadata(record: Optional[Dict]) -> np.ndarray:
    """
    Convert a metadata dict into a fixed-length numeric vector (7-dim).
    Real images (no metadata) return a zero vector with has_metadata=0.
    """
    if record is None:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    finding_map = {"none": 0, "few": 1, "moderate": 2, "many": 3}
    anat = record.get("anatomical_findings", {})
    micro = finding_map.get(anat.get("microaneurysms", "none"), 0)
    hemo = finding_map.get(anat.get("hemorrhages", "none"), 0)
    exud = finding_map.get(anat.get("exudates", "none"), 0)
    age_norm = float(record.get("age", 50)) / 120.0
    quality = float(record.get("image_quality", 0.5))
    left_eye = 1.0 if record.get("left_eye", True) else 0.0
    return np.array([micro, hemo, exud, age_norm, quality, left_eye, 1.0], dtype=np.float32)


class DRDataset(Dataset):
    """Dataset for Diabetic Retinopathy classification."""

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        metadata: Optional[pd.DataFrame] = None,
        transform=None,
        return_metadata: bool = False
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.metadata = metadata
        self.transform = transform
        self.return_metadata = return_metadata

        # Verify paths exist
        valid_indices = []
        for i, path in enumerate(image_paths):
            if Path(path).exists():
                valid_indices.append(i)
            else:
                logger.warning(f"Image not found: {path}")
        if len(valid_indices) < len(image_paths):
            logger.warning(f"Filtering to {len(valid_indices)} valid images")
            self.image_paths = [self.image_paths[i] for i in valid_indices]
            self.labels = [self.labels[i] for i in valid_indices]
            if self.metadata is not None:
                self.metadata = self.metadata.iloc[valid_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            if img_path.endswith('.npy'):
                img = np.load(img_path, allow_pickle=False)
            else:
                img = Image.open(img_path).convert('RGB')
                img = np.array(img, dtype=np.float32) / 255.0
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[-1] == 4:
                img = img[:, :, :3]
            img = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
            img = torch.zeros(3, 128, 128, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.return_metadata:
            # Real images have no metadata -> zero vector + flag 0
            meta_vec = encode_metadata(None)
            return img, label, torch.from_numpy(meta_vec)
        return img, label


class SyntheticDataset(Dataset):
    """
    Dataset combining real and synthetic data, always returning a 3-tuple
    (img, label, meta) so that DataLoader collation works uniformly.
    """

    def __init__(
        self,
        real_dataset: DRDataset,
        synthetic_images: List[np.ndarray],
        synthetic_labels: List[int],
        synthetic_metadata: List[Dict]
    ):
        self.real_dataset = real_dataset
        self.synthetic_images = synthetic_images
        self.synthetic_labels = synthetic_labels
        self.synthetic_metadata = synthetic_metadata
        self.total_real = len(real_dataset)
        self.total_synthetic = len(synthetic_images)

    def __len__(self):
        return self.total_real + self.total_synthetic

    def __getitem__(self, idx):
        if idx < self.total_real:
            # Real sample: inner dataset may return 2 or 3 values depending on
            # its return_metadata flag. Normalize to always return 3.
            sample = self.real_dataset[idx]
            if len(sample) == 3:
                img, label, meta_vec = sample
                return img, label, meta_vec
            else:
                img, label = sample
                meta_vec = torch.from_numpy(encode_metadata(None))
                return img, label, meta_vec
        else:
            synth_idx = idx - self.total_real
            img = torch.from_numpy(
                self.synthetic_images[synth_idx].copy()
            ).permute(2, 0, 1).float()
            label = torch.tensor(self.synthetic_labels[synth_idx], dtype=torch.long)
            record = None
            if self.synthetic_metadata is not None and synth_idx < len(self.synthetic_metadata):
                record = self.synthetic_metadata[synth_idx]
            meta_vec = torch.from_numpy(encode_metadata(record))
            return img, label, meta_vec