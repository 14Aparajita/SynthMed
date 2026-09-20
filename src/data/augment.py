import torch
import numpy as np
from typing import Optional

class SimpleRetinalAugmentation:
    def __init__(
        self,
        brightness_range: float = 0.1,
        contrast_range: float = 0.1,
        rotation_degrees: float = 10.0,
        horizontal_flip_prob: float = 0.5,
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.rotation_degrees = rotation_degrees
        self.horizontal_flip_prob = horizontal_flip_prob
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.brightness_range > 0:
            brightness_factor = 1.0 + np.random.uniform(-self.brightness_range, self.brightness_range)
            img = img * brightness_factor
        if self.contrast_range > 0:
            contrast_factor = 1.0 + np.random.uniform(-self.contrast_range, self.contrast_range)
            mean = img.mean()
            img = (img - mean) * contrast_factor + mean
        img = torch.clamp(img, 0.0, 1.0)
        if np.random.random() < self.horizontal_flip_prob:
            img = torch.flip(img, dims=[-1])
        return img

def get_augmentation_pipeline(strength: str = "default"):
    if strength == "heavy":
        return SimpleRetinalAugmentation(
            brightness_range=0.3,
            contrast_range=0.3,
            rotation_degrees=30,
            horizontal_flip_prob=0.5,
        )
    else:
        return SimpleRetinalAugmentation()