# SynthMed: Schema-Enforced Synthetic Medical Data Generation for Low-Resource Diabetic Retinopathy Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Medical imaging classifiers require large labeled datasets. In low-resource clinical settings, only 100–200 labeled fundus images may be available. **SynthMed** is a dual-modality synthetic data generator that produces paired, JSON-schema-valid clinical metadata and 128×128 retinal fundus images for diabetic retinopathy (DR) classification.

The system combines:

- **Retrieval-augmented generation (RAG)** for grounding synthetic metadata in real clinical knowledge
- **JSON Schema enforcement** with a bounded, rule-based repair engine
- **Lightweight denoising diffusion (DDPM)** for synthetic fundus image generation
- **Multi-seed statistical evaluation** to measure downstream utility

**Key finding:** SynthMed achieves perfect structural reliability (100% schema validity, 100% repair success) but does **not** produce statistically significant classification gains at N=100 real images (multi-seed paired t-test: p = 0.214). We characterize this *validity–utility gap* and trace it to distributional mismatch between generated and real metadata.

## Contributions

1. A dual-modality synthetic medical data generator that produces schema-valid clinical metadata paired with synthetic fundus images.
2. A bounded rule-based repair engine that achieves 100% repair success on injected corruption across four error classes.
3. A multi-seed empirical evaluation (5 seeds) demonstrating that structural validity does not imply downstream utility.
4. A distributional fidelity analysis quantifying the mismatch between generated and real metadata.

## Method

The SynthMed pipeline operates in five stages:

1. **Preprocessing** — APTOS 2019 fundus images resized to 128×128, CLAHE-enhanced, saved as `.npy`.
2. **RAG Retrieval** — A clinical knowledge base (10 documents) is embedded with `all-MiniLM-L6-v2` and indexed with FAISS. At generation time, the top-5 documents are retrieved via RAG Fusion (semantic + keyword + clinical concept search).
3. **Metadata Generation** — `distilgpt2` generates a JSON clinical record conditioned on a target DR grade and the retrieved context. Output is parsed as JSON.
4. **Schema Validation & Repair** — Each record is validated against a JSON Schema (Draft 7). Invalid records trigger a three-pass rule-based repair engine (structural → type → constraint), bounded at 3 iterations.
5. **Image Generation** — A lightweight DDPM (~5M params, T=100) is trained for 20 epochs on the training images and sampled to produce 32×32 patches upscaled to 128×128.

The classifier (MobileNetV2) is trained on combinations of real and synthetic data and evaluated on a fixed 100-image test set.

## Repository Structure

```text
SynthMed/
├── config/                         # Experiment YAML configs
│   └── schema/                     # JSON Schema for clinical metadata
├── experiments/                    # Core pipeline (run_pipeline.py)
├── src/                            # Source code
│   ├── classifier/                 # MobileNetV2 classifier and trainer
│   ├── data/                       # Preprocessing, datasets, augmentation
│   ├── evaluation/                 # Metrics and reporting
│   ├── generation/                 # DDPM, metadata generator, grounding
│   ├── retrieval/                  # FAISS index, embeddings, RAG Fusion
│   ├── schema/                     # Validator and repairer
│   └── utils/                      # Config, logging, seeding
├── scripts/                        # Utility scripts
├── tests/                          # Unit tests
├── run_caisc_experiments.py        # Final experiment runner
├── run_multiseed.py                # Multi-seed runner
└── iclr_results/                   # Final results package
    ├── figures/
    ├── metrics/
    └── tables/
```

## Environment Setup

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (optional; CPU fallback supported)

### Installation

```bash
git clone https://github.com/14Aparajita/SynthMed.git
cd SynthMed
python -m venv venv
```

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Dataset

The experiments use the **APTOS 2019 Blindness Detection** dataset, publicly available on Kaggle.

```bash
kaggle competitions download -c aptos2019-blindness-detection
```

### Data Preprocessing

Place the downloaded images in `data/raw/` and run:

```python
from src.data.preprocess import preprocess_images

preprocess_images(
    "data/raw",
    "data/processed",
    128
)
```

The pipeline expects `data/raw/clinical.csv` with the following columns:

- `image_id`
- `image_path`
- `dr_grade`
- `split`

A helper script generates a fixed stratified test set of 100 images:

```text
data/processed/fixed_test_ids.csv
```

### Knowledge Base

A default 10-document clinical knowledge base is bundled in `run_pipeline.py`.

To build a custom knowledge base:

```bash
python scripts/build_kb.py
```

## Running the Project

### Full Single-Seed Experiment Suite

```bash
python run_caisc_experiments.py
```

Runs A1–A8, B1, B2, C1 and saves results to:

```text
outputs/results/caisc_final_results_<timestamp>.csv
```

### Multi-Seed Evaluation (A1, A2, A8)

```bash
python run_multiseed.py
```

Runs seeds 42–46 for A1, A2, and A8.

Outputs:

```text
outputs/results/multiseed_summary.csv
outputs/results/multiseed_raw.csv
```

### Reliability Analysis

```bash
python scripts/save_synthetic_metadata.py
python scripts/distributional_fidelity.py
```

Generates:

```text
outputs/results/synthetic_metadata_A2.jsonl
iclr_results/metrics/distributional_fidelity.csv
```

### Figure and Table Generation

```bash
python iclr_results/figures/generate_pipeline_diagram.py
python iclr_results/figures/generate_final_figures.py
python iclr_results/figures/generate_distributional_figure.py
python scripts/build_final_table.py
```

## Experiments

The final experiment set:

| ID | Configuration | Real | Synthetic |
|---|---|---:|---:|
| A1 | Real-only baseline | 100 | 0 |
| A2 | SynthMed (full) | 100 | 500 |
| A3 | No repair | 100 | 500 |
| A4 | No RAG | 100 | 500 |
| A5 | Image-only diffusion | 100 | 500 images |
| A6 | Conditional DDPM | 100 | 500 |
| A8 | Heavy geometric augmentation | 100 | 0 |
| B1 | Baseline 200 | 200 | 0 |
| B2 | SynthMed 200 | 200 | 500 |
| C1 | Upper bound | 2000 | 0 |
| A2+fusion | Metadata late-fusion | 100 | 500 |

## Results

### Reliability

| Metric | Value | n |
|---|---:|---:|
| Schema validity rate | **1.000** | 500 |
| Repair success rate (injected corruption) | **1.000** | 100 |

### Classification (Multi-Seed, 5 seeds)

| Configuration | Accuracy (mean ± std) | F1 | ROC-AUC |
|---|---:|---:|---:|
| A1 (real only) | 0.678 ± 0.018 | 0.638 | 0.877 |
| A2 (SynthMed) | 0.628 ± 0.092 | 0.579 | 0.818 |
| A8 (geometric aug) | 0.672 ± 0.022 | 0.635 | 0.886 |

### Paired t-tests

- A2 vs A1: t = −1.474, **p = 0.214** (not significant)
- A8 vs A1: t = −0.408, **p = 0.704** (not significant)

### Distributional Fidelity

| Field | JS Divergence |
|---|---:|
| Age | **0.70** |
| Image quality | **0.83** |

### Metadata Fusion

A2 with metadata late-fusion achieves **0.610 accuracy**, compared with **0.628** for image-only A2. Metadata fusion does not improve downstream performance at N=100.

## Reproducibility

- **Seeds:** 42 (single-seed), 42–46 (multi-seed)
- **Test set:** Fixed stratified 100 images (`data/processed/fixed_test_ids.csv`)
- **Configs:** All hyperparameters in `config/exp_*.yaml`
- **Outputs:** `outputs/results/` and `iclr_results/`
- **Determinism:** `set_seed()` is called at the start of every pipeline run

## Figures and Tables

| Artifact | Path |
|---|---|
| Pipeline diagram | `iclr_results/figures/fig_pipeline.png` |
| Reliability bar chart | `iclr_results/figures/fig_reliability.png` |
| Multi-seed accuracy | `iclr_results/figures/fig_multiseed.png` |
| Distributional mismatch | `iclr_results/figures/fig_distributional_mismatch.png` |
| Main results table | `iclr_results/tables/main_results.csv` |

## Limitations

- Single dataset (APTOS 2019); no cross-dataset generalization.
- Grounding score was not reliably recorded (logged as 0.0 in all runs).
- Repair engine was never triggered on real generations (100% first-pass valid).
- Synthetic images are low-resolution (32×32 upscaled to 128×128).
- No image fidelity metric (FID/KID) was computed.
- Metadata fusion degrades performance at N=100.

## Citation

```bibtex
@misc{vaish2026synthmed,
  title={SynthMed: Schema-Enforced Synthetic Medical Data Generation for Low-Resource Diabetic Retinopathy Classification},
  author={Vaish, Aparajita},
  year={2026},
  howpublished={\url{https://github.com/14Aparajita/SynthMed}}
}
```

*Note: Update this citation when the paper is published.*

## Author

**Aparajita Vaish**

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.