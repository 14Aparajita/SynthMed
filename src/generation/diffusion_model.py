import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger("synthmed.generation")

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embeddings for timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

class UNetBlock(nn.Module):
    """Simplified UNet block for diffusion model."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(F.silu(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.norm2(F.silu(self.conv2(h)))
        return h

class LightweightDiffusion(nn.Module):
    """
    Lightweight DDPM for 32x32 medical image generation.
    Can be upscaled to 128x128 via bilinear interpolation.
    """
    
    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 3,
        base_channels: int = 64,
        time_emb_dim: int = 128,
        num_timesteps: int = 100
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_timesteps = num_timesteps
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # UNet encoder
        self.down1 = UNetBlock(in_channels, base_channels, time_emb_dim)
        self.down2 = UNetBlock(base_channels, base_channels * 2, time_emb_dim)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = UNetBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        
        # UNet decoder
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = UNetBlock(base_channels * 2, base_channels, time_emb_dim)
        self.final = nn.Conv2d(base_channels, in_channels, 1)
        
        # Noise schedule
        self.register_buffer(
            'betas',
            self._linear_beta_schedule(num_timesteps)
        )
        alphas = 1.0 - self.betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer(
            'sqrt_one_minus_alphas_cumprod',
            torch.sqrt(1.0 - self.alphas_cumprod)
        )
    
    def _linear_beta_schedule(self, timesteps: int) -> torch.Tensor:
        """Linear noise schedule."""
        beta_start = 1e-4
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Predict noise at given timesteps."""
        # Time embedding
        t_emb = self.time_embed(timesteps)
        
        # Encoder
        h1 = self.down1(x, t_emb)
        h2 = self.down2(self.pool(h1), t_emb)
        
        # Bottleneck
        h = self.bottleneck(h2, t_emb)
        
        # Decoder with skip connections
        h = self.up1(h)
        h = torch.cat([h, h1], dim=1)
        h = self.dec1(h, t_emb)
        
        return self.final(h)
    
    def add_noise(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to images according to schedule."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        while sqrt_alpha.dim() < x_start.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        x_noisy = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        
        return x_noisy, noise
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        device: str = "cpu",
        progress: bool = False
    ) -> torch.Tensor:
        """Generate images using DDPM sampling."""
        self.eval()
        x = torch.randn(batch_size, self.in_channels, self.image_size, self.image_size)
        x = x.to(device)
        
        timesteps = reversed(range(self.num_timesteps))
        iterator = tqdm(timesteps, desc="Sampling") if progress else timesteps
        
        for t in iterator:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self(x, t_batch)
            
            # Get schedule values
            alpha = 1.0 - self.betas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            # Compute coefficients
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            coef1 = 1.0 / torch.sqrt(alpha)
            coef2 = beta / torch.sqrt(1.0 - alpha_cumprod)
            
            x = coef1 * (x - coef2 * predicted_noise) + torch.sqrt(beta) * noise
        
        # Clamp to valid range
        x = torch.clamp(x, 0.0, 1.0)
        
        return x
    
    def upscale(
        self,
        images: torch.Tensor,
        target_size: int = 128
    ) -> torch.Tensor:
        """Upscale images to target size."""
        return F.interpolate(
            images,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'image_size': self.image_size,
            'in_channels': self.in_channels,
            'num_timesteps': self.num_timesteps,
        }, path)
        logger.info(f"Saved checkpoint to {path}")
    
    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> 'LightweightDiffusion':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            image_size=checkpoint['image_size'],
            in_channels=checkpoint['in_channels'],
            num_timesteps=checkpoint['num_timesteps']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model

# Add tqdm import for sampling
from tqdm import tqdm