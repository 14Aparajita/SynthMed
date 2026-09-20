# SynthMed ICLR Experiment Results

## Environment
- Python 3.10
- PyTorch 2.x with CUDA (if GPU used)
- See `requirements.txt` in the main repository.

## Dataset
- APTOS 2019 Blindness Detection (preprocessed)
- Fixed test set: 100 images (`data/processed/fixed_test_ids.csv`)
- Training subsets: 100, 200, 2000 real images (stratified)
- Full dataset: `data/raw/clinical_full.csv`

## Reproducing the experiments
1. Install dependencies:
   `pip install -r requirements.txt`
2. Ensure `data/raw/clinical_full.csv` exists. If not, create it from the original full dataset.
3. Run the full experiment suite:
   `python run_caisc_experiments.py`
4. Run multi‑seed (optional):
   `python run_multiseed.py`
5. Generate figures and tables:
   `python iclr_results/figures/generate_figures.py`
   `python iclr_results/tables/make_table.py`

## Random seeds
- Main experiments: seed = 42
- Multi‑seed: 42, 43, 44, 45, 46 (if run)

## Metrics
- Accuracy, weighted F1, weighted ROC-AUC.
- Schema validity rate and repair success rate (reported in pipeline logs).

## Output map
- `metrics/final_results.csv` – main results table
- `figures/main/fig_main_performance.png` – main performance bar chart
- `tables/results_summary.csv` and `.md` – publication-ready table
- `checkpoints/` – saved model checkpoints
- `configs/` – all experiment configs used
- `logs/` – stdout/stderr logs (if saved)

## Known issues
- Some runs may fail with "paging file too small" if system virtual memory is insufficient. Increase Windows paging file to at least 16 GB.

## Multi‑seed evaluation (A1 vs A2)
- Seeds: 42, 43, 44, 45, 46
- Mean accuracy (std):
  - A1: 0.678 (0.018)
  - A2: 0.628 (0.092)
- Paired t‑test: t = -1.474, p = 0.214
- Conclusion: No statistically significant improvement from SynthMed at N=100 real images.
- Figure: `figures/main/fig_multiseed_accuracy.png`