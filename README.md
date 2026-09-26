# SynthMed: Schema-Enforced Synthetic Medical Data Generation for Low-Resource Diabetic Retinopathy Classification

## Overview

SynthMed is a dual-modality synthetic data generator for diabetic retinopathy (DR) classification. It produces paired JSON-schema-valid clinical metadata and 128×128 fundus images, then evaluates whether those synthetic samples improve downstream classification in a low-resource regime (100 real training images).

The system combines:

- Retrieval-augmented generation (RAG) with FAISS and MiniLM-L6-v2 to ground synthetic metadata in clinical text.
- JSON Schema enforcement with a bounded, rule-based repair engine (max 3 iterations).
- Lightweight denoising diffusion (DDPM, ~5M params, T=100) for synthetic fundus images at native 128×128.
- Multi-seed statistical evaluation on a fixed 733-image test set.

## Key Finding

SynthMed achieves perfect structural reliability (schema validity = 1.000 on a 10-document knowledge base; 0.966 on a 62-document one; repair success = 0.82 on the harder corpus) but does not significantly improve DR classification at N=100 real images.

- Multi-seed paired t-test (3 seeds): SynthMed vs. real-only p = 0.3066.
- Heavy geometric augmentation vs. real-only p = 0.6195.
- FID between real and synthetic images: 22.93.
- Jensen–Shannon divergence (real vs. synthetic metadata): 0.70 (age), 0.83 (image quality).

We trace the gap to distributional mismatch and characterize it as a validity–utility gap.

## Contributions

- A dual-modality synthetic medical data generator producing schema-valid metadata paired with synthetic fundus images.
- A bounded rule-based repair engine that achieves 100% repair success on injected corruption and 79–84% success on natural LLM errors under a 62-document grounding corpus.
- A multi-seed empirical study showing that structural validity does not imply downstream utility.
- Distributional-fidelity and FID evidence identifying distributional mismatch as the mechanism.

## Method

Pipeline in execution order:

1. **Preprocessing** — APTOS 2019 fundus images resized to 128×128, CLAHE-enhanced, saved as `.npy`.
2. **RAG retrieval** — A clinical knowledge base is embedded with all-MiniLM-L6-v2 and indexed with FAISS. The top-k documents are retrieved via RAG Fusion (semantic + keyword + clinical concept search).
3. **Metadata generation** — distilgpt2 generates a JSON clinical record conditioned on a target DR grade and the retrieved context.
4. **Schema validation and repair** — Records are validated against a JSON Schema (Draft 7). Invalid records trigger a three-pass rule-based repair engine (structural → type → constraint), bounded at 3 iterations.
5. **Image generation** — A lightweight DDPM (~5M params, base_channels=32, T=100) is trained for 20 epochs on the training images and sampled in chunks of 10 to produce 32×32 patches upscaled to 128×128.
6. **Classifier training** — MobileNetV2 is trained on combinations of real and synthetic data with mixup, label smoothing, and early stopping.
7. **Evaluation** — Accuracy, weighted F1, weighted ROC-AUC, expected calibration error (ECE), per-class metrics, confusion matrix, FID.

A full pipeline diagram is at `iclr_results/figures/fig_pipeline.png`.

## Repository Structure

```text
SynthMed/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── config/
│   └── schema/
├── experiments/
│   └── run_pipeline.py
├── src/
│   ├── classifier/
│   ├── data/
│   ├── evaluation/
│   ├── generation/
│   ├── retrieval/
│   ├── schema/
│   └── utils/
├── scripts/
│   ├── build_kb.py
│   ├── build_extended_kb.py
│   ├── download_data.sh
│   ├── save_synthetic_metadata.py
│   ├── distributional_fidelity.py
│   ├── compute_fid_offline.py
│   ├── build_final_table.py
│   ├── prepare_ratio_configs.py
│   ├── summarize_ratio_sweep.py
│   └── run_all.sh
├── tests/
├── run_caisc_experiments.py
├── run_multiseed.py
├── run_ratio_sweep.ps1
├── finalize_table.ps1
└── iclr_results/
    ├── figures/
    ├── metrics/
    └── tables/
```

## Environment Setup

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (optional; CPU fallback supported, ~5× slower)

### Installation

```bash
git clone https://github.com/14Aparajita/SynthMed.git
cd SynthMed
python -m venv venv
```

**Windows:**

```bat
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Dataset

The experiments use the APTOS 2019 Blindness Detection dataset, publicly available on Kaggle.

```bash
kaggle competitions download -c aptos2019-blindness-detection
```

### Data Preparation

Place the downloaded images in:

```text
data/raw/
```

The pipeline expects:

```text
data/raw/clinical.csv
```

with the following columns:

```text
image_id
image_path
dr_grade
split
```

A full copy without the test/train split should be saved as:

```text
data/raw/clinical_full.csv
```

The runner generates a fixed stratified test set containing approximately 20% of the dataset (~733 images) and saves it to:

```text
data/processed/fixed_test_ids.csv
```

## Knowledge Base

A default 10-document clinical knowledge base is bundled inside run_pipeline.py.

To build the extended 62-document version:

```bash
python scripts/build_extended_kb.py
```

## Running the Project

### Full Single-Seed Suite

Runs experiments A1–A8, B1, B2, and C1:

```bash
python run_caisc_experiments.py
```

### Multi-Seed Evaluation

Runs A1, A2, and A8 using three seeds:

```bash
python run_multiseed.py
```

### Synthetic-to-Real Ratio Sweep

Evaluates 0, 200, 500, and 1000 synthetic samples.

On Windows, subprocess isolation is recommended to reduce paging-file OOM issues:

```powershell
python scripts/prepare_ratio_configs.py
.\run_ratio_sweep.ps1
```

Alternatively, run each configuration manually:

```bash
python experiments/run_pipeline.py --config config/exp_ratio_0.yaml
python experiments/run_pipeline.py --config config/exp_ratio_200.yaml
python experiments/run_pipeline.py --config config/exp_ratio_500.yaml
python experiments/run_pipeline.py --config config/exp_ratio_1000.yaml
```

```bash
python scripts/summarize_ratio_sweep.py
```

### Reliability and Fidelity Analysis

Generate and analyze 500 synthetic metadata records:

```bash
python scripts/save_synthetic_metadata.py
python scripts/distributional_fidelity.py
python scripts/compute_fid_offline.py
```

These scripts evaluate:

- Schema validity
- Repair success
- Jensen–Shannon divergence
- Chi-square statistics
- FID

### Figures and Tables

Generate the paper figures and final results table:

```bash
python iclr_results/figures/generate_pipeline_diagram.py
python iclr_results/figures/generate_final_figures.py
python iclr_results/figures/generate_distributional_figure.py
python scripts/build_final_table.py
```

## Experiments

| ID | Configuration | Real | Synthetic |
|---|---|---:|---:|
| A1 | Real-only baseline | 100 | 0 |
| A2 | SynthMed (full pipeline) | 100 | 500 |
| A3 | No schema repair | 100 | 500 |
| A4 | No RAG grounding | 100 | 500 |
| A5 | Image-only diffusion | 100 | 500 images |
| A6 | Class-conditional DDPM | 100 | 500 |
| A8 | Heavy geometric augmentation | 100 | 0 |
| B1 | Baseline 200 real | 200 | 0 |
| B2 | SynthMed 200 real | 200 | 500 |
| C1 | Upper bound | 2000 | 0 |
| A2-fusion | Metadata late-fusion classifier | 100 | 500 |
| A2-hightemp | Repair ablation (temperature 1.2) | 100 | 500 |
| Ratio sweep | 0, 200, 500, 1000 synthetic | 100 | Variable |

## Results

### Multi-Seed Classification

Three seeds evaluated on the fixed 733-image test set.

| Configuration | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| A1 (real only, 100) | 0.7035 ± 0.0185 | 0.6610 ± 0.0158 | 0.8940 ± 0.0041 |
| A2 (SynthMed, 100 + 500) | 0.6571 ± 0.0421 | 0.6259 ± 0.0399 | 0.8481 ± 0.0213 |
| A8 (heavy geo aug, 100) | 0.7094 ± 0.0027 | 0.6692 ± 0.0082 | 0.8906 ± 0.0135 |

#### Paired t-tests

- A2 vs A1: t = −1.3611, p = 0.3066
- A8 vs A1: t = 0.5820, p = 0.6195

### Reliability

| Metric | Value | n |
|---|---:|---:|
| Schema validity (10-doc KB) | 1.000 | 500 |
| Schema validity (62-doc KB) | 0.966 | 500 |
| Repair success (62-doc KB) | 0.824 | 17 |
| Repair success (injected corruption) | 1.000 | 100 |

### Generative Quality and Distributional Fidelity

| Metric | Value |
|---|---:|
| FID (20 real vs 20 synthetic, 128×128) | 22.93 |
| JS divergence (age) | 0.70 |
| JS divergence (image quality) | 0.83 |

### Synthetic-to-Real Ratio Sweep

| n_synth | Accuracy | F1 | ROC-AUC | Schema Valid | Repair Success |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.6862 | 0.6429 | 0.8952 | 0.000 | 0.000 |
| 200 | 0.6439 | 0.6130 | 0.8748 | 0.905 | 0.842 |
| 500 | 0.6944 | 0.6556 | 0.8819 | 0.966 | 0.824 |
| 1000 | 0.6712 | 0.6328 | 0.8234 | 0.939 | 0.787 |

No monotonic improvement with synthetic count.

## Reproducibility

- Seeds: 42 for single-seed experiments; 42–44 for multi-seed experiments; 42 fixed for the ratio sweep.
- Test set: `data/processed/fixed_test_ids.csv` (733 images, stratified).
- Configurations: Every hyperparameter is specified in a YAML file under config/.
- Determinism: set_seed() is called at the start of every pipeline run.
- Outputs: Metrics JSON files are written under `outputs/results/`; the paper package is stored under `iclr_results/`.

## Figures and Tables

| Artifact | Path |
|---|---|
| Pipeline diagram | `iclr_results/figures/fig_pipeline.png` |
| Reliability bar chart | `iclr_results/figures/fig_reliability.png` |
| Multi-seed accuracy with error bars | `iclr_results/figures/fig_multiseed.png` |
| Distributional mismatch | `iclr_results/figures/fig_distributional_mismatch.png` |
| Main paper table | `iclr_results/tables/main_results.csv` |
| Ratio sweep | `outputs/results/ratio_sweep.csv` |

## Limitations

- Single dataset (APTOS 2019); no cross-dataset generalization.
- Single LLM (DistilGPT-2) and single embedding model (MiniLM-L6-v2).
- No clinician review of synthetic images.
- FID computed on 20 samples per side due to 4 GB GPU memory.
- The extended knowledge base (62 documents) reduces schema validity relative to the 10-document default; this is reported as a finding, not a bug.
- Only 3 seeds were used for the multi-seed evaluation; more seeds would tighten the confidence intervals.

## Citation

```bibtex
@misc{vaish2026synthmed,
  title={SynthMed: Schema-Enforced Synthetic Medical Data Generation for Low-Resource Diabetic Retinopathy Classification},
  author={Vaish, Aparajita},
  year={2026},
  howpublished={\url{https://github.com/14Aparajita/SynthMed}}
}
```

Update this citation entry once the paper is published.

## License

MIT — see LICENSE for details.