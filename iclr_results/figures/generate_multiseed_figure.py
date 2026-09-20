import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("iclr_results/figures/main", exist_ok=True)

# Read the two-level header correctly
summary = pd.read_csv(
    "iclr_results/metrics/multiseed/multiseed_summary.csv",
    header=[0, 1],
    index_col=0
)

# The first column header is stored as ('Unnamed: 0', 'experiment')
summary.index.name = "experiment"

experiments = [
    "A1_baseline_100real",
    "A2_synthmed_100real_500syn"
]

means = [
    summary.loc[exp, ("accuracy", "mean")]
    for exp in experiments
]

stds = [
    summary.loc[exp, ("accuracy", "std")]
    for exp in experiments
]

labels = [
    "Baseline (100 real)",
    "SynthMed (100+500)"
]

fig, ax = plt.subplots(figsize=(6, 5))

ax.bar(
    labels,
    means,
    yerr=stds,
    capsize=5,
    color=["#3498DB", "#E67E22"],
    alpha=0.85
)

ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.set_title("Multi-seed Accuracy (5 seeds)")

# Existing reported statistical annotation
ax.text(
    0.5,
    0.75,
    "p = 0.214 (paired t-test)",
    ha="center",
    fontsize=10
)

plt.tight_layout()

output_path = "iclr_results/figures/main/fig_multiseed_accuracy.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved multiseed accuracy figure: {output_path}")