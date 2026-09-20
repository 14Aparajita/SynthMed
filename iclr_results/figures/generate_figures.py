# iclr_results/figures/generate_figures.py
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("iclr_results/figures/main", exist_ok=True)

df = pd.read_csv("iclr_results/metrics/final_results.csv")
# Keep only main experiments
include = [
    "A1_baseline_100real",
    "A2_synthmed_100real_500syn",
    "A3_norepair_100real_500syn",
    "A4_norag_100real_500syn",
    "A5_imageonly_diffusion",
    "A8_geometric_only",
    "A6_synthmed_conditional",
    "B1_baseline_200real",
    "B2_synthmed_200real_500syn",
    "C1_upperbound_fulldata",
]
df = df[df["experiment"].isin(include)].copy()
order = {name: i for i, name in enumerate(include)}
df["order"] = df["experiment"].map(order)
df = df.sort_values("order")

fig, ax = plt.subplots(figsize=(12, 6))
x = range(len(df))
width = 0.35
ax.bar([i - width/2 for i in x], df["accuracy"], width, label="Accuracy", alpha=0.85)
ax.bar([i + width/2 for i in x], df["f1_score"], width, label="F1", alpha=0.85)
ax.set_xticks(list(x))
ax.set_xticklabels(df["experiment"], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Score")
ax.set_title("SynthMed Performance Comparison")
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("iclr_results/figures/main/fig_main_performance.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved main performance figure")