"""
src/generation/image_gen.py

Synthetic retinal image generation for SynthMed.

Architecture:
  - Beta-VAE: encodes real retinal patches into a structured latent space,
    conditioned on clinical metadata (DR grade, severity score).
  - Lightweight DDPM: denoising diffusion in latent space for diverse synthesis.
  - Metadata conditioning: embeds class label + severity into latent conditioning.

Designed to be laptop-runnable:
  - Image size: 224x224 (or 64x64 for fast experiments)
  - VAE latent dim: 128
  - Diffusion steps: 50 (inference), 200 (training)
  - Full training <30 min on CPU for 500 images
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)


# ── Utility modules ────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res = ResBlock(out_ch)

    def forward(self, x):
        return self.res(self.conv(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res = ResBlock(out_ch)

    def forward(self, x):
        return self.res(self.conv(x))


# ── Beta-VAE ─────────────────────────────────────────────────────────

class BetaVAE(nn.Module):
    """
    Beta-VAE for retinal image synthesis.
    Conditioned on metadata embedding (class + severity).
    Latent dim = 128, input = 3×224×224 (or 3×64×64 for fast mode).
    """

    def __init__(
        self,
        image_size: int = 64,
        latent_dim: int = 128,
        n_classes: int = 2,
        cond_embed_dim: int = 64,
        beta: float = 4.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.beta = beta

        base_ch = 32
        # Conditioning embedding
        self.cond_embed = nn.Sequential(
            nn.Embedding(n_classes, cond_embed_dim // 2),
        )
        self.severity_embed = nn.Linear(1, cond_embed_dim // 2)
        cond_dim = cond_embed_dim

        # Encoder: 3×H×W → latent
        self.encoder = nn.Sequential(
            DownBlock(3, base_ch),           # H/2
            DownBlock(base_ch, base_ch * 2), # H/4
            DownBlock(base_ch * 2, base_ch * 4),  # H/8
            DownBlock(base_ch * 4, base_ch * 8),  # H/16
        )
        enc_spatial = image_size // 16
        enc_flat = base_ch * 8 * enc_spatial * enc_spatial
        self.enc_fc = nn.Linear(enc_flat + cond_dim, 512)
        self.mu_head = nn.Linear(512, latent_dim)
        self.logvar_head = nn.Linear(512, latent_dim)

        # Decoder: latent + cond → 3×H×W
        dec_spatial = image_size // 16
        self.dec_fc = nn.Linear(latent_dim + cond_dim, base_ch * 8 * dec_spatial * dec_spatial)
        self.dec_spatial = dec_spatial
        self.dec_ch = base_ch * 8
        self.decoder = nn.Sequential(
            UpBlock(base_ch * 8, base_ch * 4),  # *2
            UpBlock(base_ch * 4, base_ch * 2),  # *4
            UpBlock(base_ch * 2, base_ch),       # *8
            UpBlock(base_ch, base_ch),            # *16
            nn.Conv2d(base_ch, 3, 1),
            nn.Tanh(),
        )

    def _cond(self, labels: torch.Tensor, severity: Optional[torch.Tensor]) -> torch.Tensor:
        class_emb = self.cond_embed(labels)  # (B, cond//2)
        if severity is None:
            severity = torch.zeros(labels.size(0), 1, device=labels.device)
        sev_emb = self.severity_embed(severity.float())  # (B, cond//2)
        return torch.cat([class_emb, sev_emb], dim=1)

    def encode(
        self, x: torch.Tensor, labels: torch.Tensor,
        severity: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cond = self._cond(labels, severity)
        h = self.encoder(x).flatten(1)
        h = torch.cat([h, cond], dim=1)
        h = F.silu(self.enc_fc(h))
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(
        self, z: torch.Tensor, labels: torch.Tensor,
        severity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        cond = self._cond(labels, severity)
        h = self.dec_fc(torch.cat([z, cond], dim=1))
        h = h.view(-1, self.dec_ch, self.dec_spatial, self.dec_spatial)
        return self.decoder(h)

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor,
        severity: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, labels, severity)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, labels, severity)
        return recon, mu, logvar

    def elbo_loss(
        self,
        recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * kl_loss
        return loss, {"recon": recon_loss.item(), "kl": kl_loss.item(), "total": loss.item()}

    @torch.no_grad()
    def sample(
        self, n: int, labels: torch.Tensor,
        severity: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z, labels, severity)


# ── Lightweight DDPM in latent space ─────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        emb = t[:, None].float() * freqs[None]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class LatentDiffusionMLP(nn.Module):
    """
    Small MLP-based denoiser operating in VAE latent space.
    Input: noisy latent z_t + time embedding + conditioning
    Output: predicted noise epsilon
    """

    def __init__(self, latent_dim: int = 128, cond_dim: int = 64, time_dim: int = 128):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(time_dim)
        self.cond_proj = nn.Linear(cond_dim, time_dim)
        hidden = 512
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_dim * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(
        self,
        z: torch.Tensor,       # (B, latent_dim)
        t: torch.Tensor,       # (B,) int timesteps
        cond: torch.Tensor,    # (B, cond_dim)
    ) -> torch.Tensor:
        t_emb = self.time_emb(t)
        c_emb = F.silu(self.cond_proj(cond))
        x = torch.cat([z, t_emb, c_emb], dim=-1)
        return self.net(x)


# ── DDPM noise schedule ───────────────────────────────────────────────

class LinearNoiseSchedule:
    def __init__(self, n_steps: int = 200, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.n_steps = n_steps
        betas = torch.linspace(beta_start, beta_end, n_steps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register(betas, alphas, alpha_cumprod)

    def register(self, betas, alphas, acp):
        self.betas = betas
        self.alphas = alphas
        self.alpha_cumprod = acp
        self.sqrt_acp = acp.sqrt()
        self.sqrt_one_minus_acp = (1.0 - acp).sqrt()

    def to(self, device):
        for attr in ("betas", "alphas", "alpha_cumprod", "sqrt_acp", "sqrt_one_minus_acp"):
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(z0)
        a = self.sqrt_acp[t].unsqueeze(-1)
        b = self.sqrt_one_minus_acp[t].unsqueeze(-1)
        return a * z0 + b * noise, noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: LatentDiffusionMLP,
        shape: tuple,
        cond: torch.Tensor,
        device: str = "cpu",
        n_inference_steps: int = 50,
    ) -> torch.Tensor:
        """DDPM reverse sampling with strided inference."""
        z = torch.randn(shape, device=device)
        step_stride = self.n_steps // n_inference_steps
        timesteps = list(reversed(range(0, self.n_steps, step_stride)))

        for t_val in timesteps:
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            eps_pred = model(z, t, cond)
            alpha = self.alphas[t_val]
            alpha_cp = self.alpha_cumprod[t_val]
            beta = self.betas[t_val]
            # DDPM update
            coef = (1 - alpha) / (1 - alpha_cp).sqrt()
            z = (1 / alpha.sqrt()) * (z - coef * eps_pred)
            if t_val > 0:
                z += beta.sqrt() * torch.randn_like(z)
        return z


# ── Image synthesizer ─────────────────────────────────────────────────

class ImageSynthesizer:
    """
    End-to-end image synthesis pipeline.
    Trains VAE on real images, then uses VAE+diffusion to generate synthetic images.
    """

    def __init__(self, cfg: dict, device: Optional[str] = None):
        self.cfg = cfg
        img_cfg = cfg["image_generation"]
        self.image_size = img_cfg["image_size"][0]
        self.latent_dim = img_cfg["latent_dim"]
        self.n_classes = 2
        self.beta = img_cfg["elbo_beta"]
        self.checkpoint_dir = Path(img_cfg["checkpoint_dir"])
        self.output_dir = Path(img_cfg["output_dir"])
        self.n_diffusion_steps_train = img_cfg["n_diffusion_steps_train"]
        self.n_diffusion_steps_infer = img_cfg["n_diffusion_steps"]

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        logger.info("ImageSynthesizer device: %s", self.device)

        cond_dim = img_cfg["conditioning"]["embed_dim"]
        self.vae = BetaVAE(
            image_size=self.image_size,
            latent_dim=self.latent_dim,
            n_classes=self.n_classes,
            cond_embed_dim=cond_dim,
            beta=self.beta,
        ).to(self.device)

        self.denoiser = LatentDiffusionMLP(
            latent_dim=self.latent_dim,
            cond_dim=cond_dim,
            time_dim=128,
        ).to(self.device)

        self.schedule = LinearNoiseSchedule(
            n_steps=self.n_diffusion_steps_train,
            beta_start=cfg["image_generation"]["beta_start"],
            beta_end=cfg["image_generation"]["beta_end"],
        ).to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def _cond_embed(self, labels: torch.Tensor, severity: Optional[torch.Tensor] = None):
        """Get conditioning embedding from VAE's cond module."""
        with torch.no_grad():
            class_emb = self.vae.cond_embed(labels)
            if severity is None:
                severity = torch.zeros(labels.size(0), 1, device=self.device)
            sev_emb = self.vae.severity_embed(severity.float())
        return torch.cat([class_emb, sev_emb], dim=1).detach()

    def pretrain_vae(self, dataloader: DataLoader, epochs: int = 20, lr: float = 1e-3):
        """Pretrain VAE on real images."""
        opt = torch.optim.Adam(self.vae.parameters(), lr=lr)
        self.vae.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataloader:
                images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                severity = batch.get("severity")
                if severity is not None:
                    severity = severity.to(self.device).unsqueeze(1)
                recon, mu, logvar = self.vae(images, labels, severity)
                loss, info = self.vae.elbo_loss(recon, images, mu, logvar)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                opt.step()
                total_loss += info["total"]
            avg = total_loss / len(dataloader)
            if (epoch + 1) % 5 == 0:
                logger.info("VAE epoch %d/%d | loss=%.4f", epoch + 1, epochs, avg)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.vae.state_dict(), self.checkpoint_dir / "vae.pt")
        logger.info("VAE saved to %s/vae.pt", self.checkpoint_dir)

    def train_diffusion(self, dataloader: DataLoader, epochs: int = 20, lr: float = 1e-4):
        """Train latent diffusion denoiser on VAE latents."""
        self.vae.eval()
        self.denoiser.train()
        opt = torch.optim.Adam(self.denoiser.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataloader:
                images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                severity = batch.get("severity")
                if severity is not None:
                    severity = severity.to(self.device).unsqueeze(1)

                with torch.no_grad():
                    mu, logvar = self.vae.encode(images, labels, severity)
                    z0 = self.vae.reparameterize(mu, logvar)

                t = torch.randint(0, self.n_diffusion_steps_train, (z0.size(0),), device=self.device)
                z_noisy, noise = self.schedule.q_sample(z0, t)
                cond = self._cond_embed(labels, severity)
                noise_pred = self.denoiser(z_noisy, t, cond)
                loss = F.mse_loss(noise_pred, noise)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()

            avg = total_loss / len(dataloader)
            if (epoch + 1) % 5 == 0:
                logger.info("Diffusion epoch %d/%d | loss=%.4f", epoch + 1, epochs, avg)

        torch.save(self.denoiser.state_dict(), self.checkpoint_dir / "diffusion.pt")
        logger.info("Diffusion model saved.")

    def load_checkpoints(self) -> bool:
        vae_ckpt = self.checkpoint_dir / "vae.pt"
        diff_ckpt = self.checkpoint_dir / "diffusion.pt"
        if vae_ckpt.exists() and diff_ckpt.exists():
            self.vae.load_state_dict(torch.load(vae_ckpt, map_location=self.device))
            self.denoiser.load_state_dict(torch.load(diff_ckpt, map_location=self.device))
            logger.info("Loaded VAE + diffusion checkpoints.")
            return True
        return False

    @torch.no_grad()
    def generate_images(
        self,
        labels: List[int],
        severities: Optional[List[float]] = None,
        output_dir: Optional[str] = None,
        save: bool = True,
    ) -> torch.Tensor:
        """Generate synthetic images for a list of class labels."""
        self.vae.eval()
        self.denoiser.eval()
        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        label_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        sev_tensor = None
        if severities is not None:
            sev_tensor = torch.tensor(severities, dtype=torch.float32, device=self.device).unsqueeze(1)

        cond = self._cond_embed(label_tensor, sev_tensor)
        # Diffusion sample in latent space
        z_sampled = self.schedule.p_sample_loop(
            model=self.denoiser,
            shape=(len(labels), self.latent_dim),
            cond=cond,
            device=self.device,
            n_inference_steps=self.n_diffusion_steps_infer,
        )
        # Decode latent to image
        images = self.vae.decode(z_sampled, label_tensor, sev_tensor)
        images = (images.clamp(-1, 1) + 1) / 2  # [0, 1]

        if save:
            to_pil = transforms.ToPILImage()
            for i, (img_tensor, lbl) in enumerate(zip(images, labels)):
                pil_img = to_pil(img_tensor.cpu())
                fname = out_dir / f"synthetic_{lbl}_{i:05d}.png"
                pil_img.save(str(fname))

        logger.info("Generated %d synthetic images → %s", len(labels), out_dir)
        return images


# ── Synthetic seed dataset (when no real data available) ──────────────

class SyntheticSeedDataset(Dataset):
    """
    Generates deterministic synthetic retinal-like images using
    parametric noise + structure for VAE bootstrapping.
    No real data required.
    """

    def __init__(
        self,
        n_samples: int = 500,
        image_size: int = 64,
        n_classes: int = 2,
        class_balance: List[float] = None,
        seed: int = 42,
    ):
        self.n_samples = n_samples
        self.image_size = image_size
        rng = np.random.RandomState(seed)
        balance = class_balance or [0.55, 0.45]
        self.labels = (rng.random(n_samples) > balance[0]).astype(np.int64)
        self.rng = rng
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def _generate_retinal_like(self, label: int, idx: int) -> np.ndarray:
        """Generate a synthetic retinal-like image patch with structure."""
        rng = np.random.RandomState(idx * 37 + label * 1000)
        H = W = self.image_size
        img = np.zeros((H, W, 3), dtype=np.float32)

        # Background: orange-red fundus color
        base_r = 0.55 + rng.uniform(-0.1, 0.1)
        base_g = 0.15 + rng.uniform(-0.05, 0.05)
        img[:, :, 0] = base_r
        img[:, :, 1] = base_g
        img[:, :, 2] = 0.05

        # Vasculature: darker branching structures
        cx, cy = W // 2, H // 2
        for _ in range(rng.randint(4, 9)):
            x0 = cx + rng.randint(-W // 4, W // 4)
            y0 = cy + rng.randint(-H // 4, H // 4)
            length = rng.randint(H // 6, H // 3)
            angle = rng.uniform(0, 2 * np.pi)
            for t in range(length):
                xi = int(x0 + t * np.cos(angle)) % W
                yi = int(y0 + t * np.sin(angle)) % H
                img[yi, xi, 0] *= 0.4
                img[yi, xi, 1] *= 0.3

        # Optic disc: bright region
        disc_x = int(cx + W * 0.2)
        disc_y = cy
        disc_r = max(3, H // 10)
        Y, X = np.ogrid[:H, :W]
        disc_mask = (X - disc_x) ** 2 + (Y - disc_y) ** 2 <= disc_r ** 2
        img[disc_mask, 0] = 0.95
        img[disc_mask, 1] = 0.92
        img[disc_mask, 2] = 0.75

        # Pathological features (for DR class)
        if label == 1:
            n_ma = rng.randint(5, 20)
            for _ in range(n_ma):
                mx = rng.randint(W // 4, 3 * W // 4)
                my = rng.randint(H // 4, 3 * H // 4)
                r = rng.randint(1, max(2, H // 40))
                mask = (X - mx) ** 2 + (Y - my) ** 2 <= r ** 2
                img[mask, 0] = 0.6
                img[mask, 1] = 0.0
                img[mask, 2] = 0.0

        # Add noise
        img += rng.normal(0, 0.02, img.shape).astype(np.float32)
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        return img

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        label = int(self.labels[idx])
        img_np = self._generate_retinal_like(label, idx)
        img_pil = Image.fromarray(img_np)
        img_t = self.transform(img_pil)
        severity = float(label) * (0.3 + self.rng.uniform(0, 0.7))
        return {
            "image": img_t,
            "label": torch.tensor(label, dtype=torch.long),
            "severity": torch.tensor(severity, dtype=torch.float32),
        }
