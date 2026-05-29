"""Test generation components."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.generation import MetadataGenerator, LightweightDiffusion

def test_metadata_generator():
    generator = MetadataGenerator(device="cpu")
    
    records = generator.generate_structured(dr_grade=2, num_records=1)
    
    assert len(records) == 1
    assert isinstance(records[0], dict)
    assert "patient_id" in records[0]

def test_diffusion_model():
    model = LightweightDiffusion(image_size=16, num_timesteps=10)
    
    # Test forward pass
    x = torch.randn(1, 3, 16, 16)
    t = torch.tensor([5])
    output = model(x, t)
    
    assert output.shape == (1, 3, 16, 16)
    
    # Test sampling
    samples = model.sample(batch_size=2, device="cpu")
    assert samples.shape == (2, 3, 16, 16)
    assert samples.min() >= 0.0
    assert samples.max() <= 1.0

def test_diffusion_upscale():
    model = LightweightDiffusion(image_size=16)
    x = torch.randn(1, 3, 16, 16)
    upscaled = model.upscale(x, target_size=32)
    
    assert upscaled.shape == (1, 3, 32, 32)