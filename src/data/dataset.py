# src/data/dataset.py - FIXED VERSION

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import logging

logger = logging.getLogger("synthmed.data")

class DRDataset(Dataset):
    """Dataset for Diabetic Retinopathy classification."""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        metadata: Optional[pd.DataFrame] = None,
        transform=None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.metadata = metadata
        self.transform = transform
        
        # Verify paths exist
        valid_indices = []
        for i, path in enumerate(image_paths):
            if Path(path).exists():
                valid_indices.append(i)
            else:
                logger.warning(f"Image not found: {path}")
        
        if len(valid_indices) < len(image_paths):
            logger.warning(f"Filtering to {len(valid_indices)} valid images")
            self.image_paths = [image_paths[i] for i in valid_indices]
            self.labels = [labels[i] for i in valid_indices]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # Try loading numpy file first
            if img_path.endswith('.npy'):
                img = np.load(img_path, allow_pickle=False)
            else:
                # Load image file (PNG, JPG, etc.)
                img = Image.open(img_path).convert('RGB')
                img = np.array(img, dtype=np.float32) / 255.0
            
            # Ensure correct shape [H, W, C]
            if len(img.shape) == 2:
                # Grayscale, convert to RGB
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[-1] == 4:
                # RGBA, remove alpha channel
                img = img[:, :, :3]
            
            # Convert to tensor [C, H, W]
            img = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
            
            if self.transform:
                img = self.transform(img)
            
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            
            return img, label
            
        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
            # Return a blank image as fallback
            img = torch.zeros(3, 128, 128, dtype=torch.float32)
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img, label

class SyntheticDataset(Dataset):
    """Dataset combining real and synthetic data."""
    
    def __init__(
        self,
        real_dataset: DRDataset,
        synthetic_images: List[np.ndarray],
        synthetic_labels: List[int],
        synthetic_metadata: pd.DataFrame
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
            return self.real_dataset[idx]
        else:
            synth_idx = idx - self.total_real
            img = torch.from_numpy(
                self.synthetic_images[synth_idx].copy()
            ).permute(2, 0, 1).float()
            label = torch.tensor(
                self.synthetic_labels[synth_idx], dtype=torch.long
            )
            return img, label