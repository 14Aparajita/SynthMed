import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging
from tqdm import tqdm

logger = logging.getLogger("synthmed.generation")


class SinusoidalPositionEmbedding(nn.Module):
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
    DDPM for fundus image generation.

    Default `base_channels=32` reduces activation memory 4x relative to 64,
    so 128x128 images fit on a 4 GB GPU (GTX 1650). Sampling is chunked
    internally to further reduce peak memory.
    """

    def __init__(
        self,
        image_size: int = 128,
        in_channels: int = 3,
        base_channels: int = 32,
        time_emb_dim: int = 128,
        num_timesteps: int = 100,
        num_classes: int = 5,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes

        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.label_embed = nn.Embedding(num_classes, time_emb_dim)

        self.down1 = UNetBlock(in_channels, base_channels, time_emb_dim)
        self.down2 = UNetBlock(base_channels, base_channels * 2, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = UNetBlock(base_channels * 2, base_channels * 2, time_emb_dim)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = UNetBlock(base_channels * 2, base_channels, time_emb_dim)
        self.final = nn.Conv2d(base_channels, in_channels, 1)

        betas = self._linear_beta_schedule(num_timesteps)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))

    def _linear_beta_schedule(self, timesteps: int) -> torch.Tensor:
        return torch.linspace(1e-4, 0.02, timesteps)

    def forward(self, x, timesteps, labels=None):
        t_emb = self.time_embed(timesteps)
        if labels is not None:
            t_emb = t_emb + self.label_embed(labels)
        h1 = self.down1(x, t_emb)
        h2 = self.down2(self.pool(h1), t_emb)
        h = self.bottleneck(h2, t_emb)
        h = self.up1(h)
        h = torch.cat([h, h1], dim=1)
        h = self.dec1(h, t_emb)
        return self.final(h)

    def add_noise(self, x_start, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps]
        while sqrt_alpha.dim() < x_start.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise, noise

    @torch.no_grad()
    def sample(self, batch_size=1, device="cpu", progress=False, labels=None, chunk_size=10):
        """
        Sample images. Internally processes in chunks of `chunk_size` to keep
        peak GPU memory bounded regardless of the requested total batch.
        """
        self.eval()
        results = []
        total_chunks = (batch_size + chunk_size - 1) // chunk_size
        chunk_iter = range(total_chunks)
        if progress:
            chunk_iter = tqdm(chunk_iter, desc="Sampling")

        for chunk_idx in chunk_iter:
            start = chunk_idx * chunk_size
            current = min(chunk_size, batch_size - start)
            chunk_labels = labels[start:start + current] if labels is not None else None

            x = torch.randn(current, self.in_channels, self.image_size, self.image_size, device=device)

            for t in reversed(range(self.num_timesteps)):
                t_batch = torch.full((current,), t, device=device, dtype=torch.long)
                predicted_noise = self(x, t_batch, labels=chunk_labels)

                alpha = 1.0 - self.betas[t]
                alpha_cumprod = self.alphas_cumprod[t]
                beta = self.betas[t]

                noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
                coef1 = 1.0 / torch.sqrt(alpha)
                coef2 = beta / torch.sqrt(1.0 - alpha_cumprod)
                x = coef1 * (x - coef2 * predicted_noise) + torch.sqrt(beta) * noise

            results.append(torch.clamp(x, 0.0, 1.0))
            del x
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        return torch.cat(results, dim=0)

    def upscale(self, images, target_size=128):
        return F.interpolate(images, size=(target_size, target_size), mode="bilinear", align_corners=False)

    def save_checkpoint(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "image_size": self.image_size,
            "in_channels": self.in_channels,
            "num_timesteps": self.num_timesteps,
            "num_classes": self.num_classes,
            "base_channels": self.base_channels,
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    @classmethod
    def load_checkpoint(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            image_size=checkpoint["image_size"],
            in_channels=checkpoint["in_channels"],
            num_timesteps=checkpoint["num_timesteps"],
            num_classes=checkpoint.get("num_classes", 5),
            base_channels=checkpoint.get("base_channels", 32),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model