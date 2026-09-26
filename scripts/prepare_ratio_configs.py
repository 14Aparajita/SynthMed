"""Generate four YAML configs for the ratio sweep. No heavy imports."""
import yaml
from pathlib import Path

REPO = Path(__file__).parent.parent
RATIOS = [0, 200, 500, 1000]

for n_syn in RATIOS:
    is_synth = n_syn > 0
    name = f"ratio_{n_syn}"
    cfg = {
        "experiment": {"name": name, "seed": 42, "device": "cuda"},
        "data": {
            "raw_dir": "data/raw", "processed_dir": "data/processed",
            "knowledge_base_dir": "data/knowledge_base", "image_size": 128,
            "num_real_train": 100, "num_real_test": 733,
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
    out = REPO / f"config/exp_{name}.yaml"
    with open(out, "w") as f:
        yaml.dump(cfg, f)
    print(f"Wrote {out}")