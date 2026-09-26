"""
Synthetic-to-real ratio sweep for SynthMed.
Runs each ratio in a separate subprocess so GPU memory is fully released.
Ratios: {0, 200, 500, 1000} at 100 real images.
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent

# Prepare shared test set once
full_df = pd.read_csv(REPO_ROOT / "data/raw/clinical_full.csv")
full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
_, test_df = train_test_split(
    full_df, test_size=0.20, stratify=full_df["dr_grade"], random_state=42
)
test_df.to_csv(REPO_ROOT / "data/processed/fixed_test_ids.csv", index=False)
test_ids = set(test_df["image_id"])
pool_df = full_df[~full_df["image_id"].isin(test_ids)]

seed = 42
tr, _ = train_test_split(
    pool_df, train_size=100, stratify=pool_df["dr_grade"], random_state=seed
)
tmp = pd.concat([tr.assign(split="train"), test_df.assign(split="test")])
tmp.to_csv(REPO_ROOT / "data/raw/clinical.csv", index=False)

ratios = [0, 200, 500, 1000]
summary_rows = []

for n_syn in ratios:
    is_synth = n_syn > 0
    name = f"ratio_{n_syn}"
    cfg_path = REPO_ROOT / f"config/exp_{name}.yaml"

    cfg = {
        "experiment": {"name": name, "seed": seed, "device": "cuda"},
        "data": {
            "raw_dir": "data/raw", "processed_dir": "data/processed",
            "knowledge_base_dir": "data/knowledge_base", "image_size": 128,
            "num_real_train": 100, "num_real_test": len(test_df),
            "num_synthetic_metadata": n_syn if is_synth else 0,
            "num_synthetic_images": n_syn if is_synth else 0,
            "augmentation_strength": "default",
        },
        "schema": {
            "schema_path": "config/schema/clinical_metadata.json",
            "repair_enabled": is_synth, "repair_max_iterations": 3,
        },
        "retrieval": {
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "top_k": 5, "fusion_weights": [0.4, 0.3, 0.3],
            "index_path": "outputs/models/faiss_index.bin",
            "rag_enabled": is_synth,
        },
        "generation": {
            "metadata_model": "distilgpt2", "metadata_max_length": 256,
            "temperature": 0.7, "diffusion_timesteps": 100,
            "diffusion_image_size": 128,
            "diffusion_checkpoint": "outputs/models/diffusion_unet.pt",
            "diffusion_epochs": 20, "conditioning_enabled": False,
        },
        "classifier": {
            "model_name": "mobilenet_v2", "num_classes": 5,
            "batch_size": 8, "epochs": 50,
            "learning_rate": 0.0001, "weight_decay": 0.001,
            "use_metadata": False,
        },
        "evaluation": {
            "metrics": ["accuracy", "f1", "roc_auc"],
            "save_results": True, "results_path": "outputs/results",
        },
    }

    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    print(f"\n{'='*60}")
    print(f"Running ratio {n_syn} in subprocess")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env["SYNTHMED_SKIP_FID"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [sys.executable, "experiments/run_pipeline.py",
           "--config", str(cfg_path.relative_to(REPO_ROOT))]
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)

    if result.returncode != 0:
        print(f"ratio {n_syn} FAILED with code {result.returncode}")
        continue

    metrics_file = REPO_ROOT / f"outputs/results/{name}_metrics.json"
    if metrics_file.exists():
        import json
        with open(metrics_file) as f:
            m = json.load(f)
        summary_rows.append({
            "n_synth": n_syn,
            "accuracy": m.get("accuracy", 0.0),
            "f1_score": m.get("f1_score", 0.0),
            "roc_auc": m.get("roc_auc", 0.0),
            "schema_validity": m.get("schema_validity_rate", 0.0),
            "repair_success": m.get("repair_success_rate", 0.0),
            "grounding": m.get("mean_grounding_score", 0.0),
        })
        print(f"\nratio {n_syn}: acc={m['accuracy']:.4f}\n")

out_path = REPO_ROOT / "outputs/results/ratio_sweep.csv"
pd.DataFrame(summary_rows).to_csv(out_path, index=False)
print(pd.DataFrame(summary_rows).to_string(index=False))
print(f"\nSaved: {out_path}")