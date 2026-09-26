"""
Offline FID computation on saved sample images.
Reads outputs/fid_samples/real/*.npy and outputs/fid_samples/synth/*.npy.

Run:   python scripts/compute_fid_offline.py
Writes: outputs/results/fid.json  (or fid_skipped.json if FID fails)

This script is designed to never crash the caller. All errors are caught.
"""

import os
import sys
import json
import traceback
from pathlib import Path

import numpy as np

OUT_JSON = Path("outputs/results/fid.json")


def log(msg):
    print(f"[FID] {msg}", flush=True)


def main():
    real_dir = Path("outputs/fid_samples/real")
    synth_dir = Path("outputs/fid_samples/synth")

    if not real_dir.exists() or not synth_dir.exists():
        log(f"Sample directories not found: {real_dir} / {synth_dir}")
        OUT_JSON.write_text(json.dumps({"fid": -1.0, "reason": "no_samples"}))
        return 0

    real_files = sorted(real_dir.glob("*.npy"))
    synth_files = sorted(synth_dir.glob("*.npy"))

    log(f"Found {len(real_files)} real and {len(synth_files)} synthetic samples")

    if len(real_files) < 2 or len(synth_files) < 2:
        log("Need at least 2 samples per side")
        OUT_JSON.write_text(json.dumps({"fid": -1.0, "reason": "insufficient_samples"}))
        return 0

    real_arr = np.stack([np.load(p) for p in real_files[:20]])
    synth_arr = np.stack([np.load(p) for p in synth_files[:20]])

    log(f"Real array: {real_arr.shape}, dtype={real_arr.dtype}, range=[{real_arr.min():.3f}, {real_arr.max():.3f}]")
    log(f"Synth array: {synth_arr.shape}, dtype={synth_arr.dtype}, range=[{synth_arr.min():.3f}, {synth_arr.max():.3f}]")

    real_u8 = (np.clip(real_arr, 0, 1) * 255).astype(np.uint8)
    synth_u8 = (np.clip(synth_arr, 0, 1) * 255).astype(np.uint8)

    try:
        import torch
        from torchmetrics.image.fid import FrechetInceptionDistance
    except Exception as e:
        log(f"Import failed: {e}")
        OUT_JSON.write_text(json.dumps({"fid": -1.0, "reason": "import_error", "error": str(e)}))
        return 0

    real_t = torch.from_numpy(real_u8).permute(0, 3, 1, 2).contiguous()
    synth_t = torch.from_numpy(synth_u8).permute(0, 3, 1, 2).contiguous()

    # Try GPU first
    fid_score = -1.0
    used_device = None

    if torch.cuda.is_available():
        try:
            log("Attempting FID on CUDA")
            torch.cuda.empty_cache()
            fid = FrechetInceptionDistance(feature=64, normalize=False).to("cuda")
            fid.update(real_t.cuda(), real=True)
            fid.update(synth_t.cuda(), real=False)
            fid_score = float(fid.compute().item())
            used_device = "cuda"
            del fid
            torch.cuda.empty_cache()
            log(f"FID (CUDA): {fid_score:.4f}")
        except Exception as e:
            log(f"CUDA FID failed: {type(e).__name__}: {e}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    if fid_score < 0:
        try:
            log("Attempting FID on CPU")
            fid = FrechetInceptionDistance(feature=64, normalize=False)
            fid.update(real_t, real=True)
            fid.update(synth_t, real=False)
            fid_score = float(fid.compute().item())
            used_device = "cpu"
            del fid
            log(f"FID (CPU): {fid_score:.4f}")
        except Exception as e:
            log(f"CPU FID failed: {type(e).__name__}: {e}")
            traceback.print_exc()

    payload = {
        "fid": fid_score,
        "device": used_device,
        "n_real": len(real_files),
        "n_synth": len(synth_files),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    log(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())