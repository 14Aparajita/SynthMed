"""
Generate the two figures used in the main paper:
  fig_reliability.png     - schema validity and repair success
  fig_multiseed.png       - multi-seed accuracy with error bars
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

out = Path("iclr_results/figures")
out.mkdir(parents=True, exist_ok=True)

# --- Figure 1: Reliability ---
fig, ax = plt.subplots(figsize=(5.2, 4))
labels = ["Schema\nvalidity", "Repair\nsuccess"]
values = [1.00, 1.00]
ns = [500, 100]
bars = ax.bar(labels, values, color=["#2E86AB", "#A23B72"], width=0.55)
for bar, v, n in zip(bars, values, ns):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
            f"{v:.2f}\n(n={n})", ha="center", fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Rate")
ax.set_title("Structured reliability of SynthMed")
plt.tight_layout()
plt.savefig(out / "fig_reliability.png", dpi=300, bbox_inches="tight")
plt.close()

# --- Figure 2: Multi-seed accuracy ---
summary = {
    "A1\n(real only)": (0.678, 0.0179),
    "A2\n(SynthMed)": (0.628, 0.0918),
    "A8\n(geo aug)": (0.672, 0.0217),
}
labels = list(summary.keys())
means = [summary[k][0] for k in labels]
stds = [summary[k][1] for k in labels]

fig, ax = plt.subplots(figsize=(6.2, 4.5))
bars = ax.bar(labels, means, yerr=stds, capsize=6,
              color=["#4C72B0", "#DD8452", "#55A868"], width=0.55)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, m + 0.015,
            f"{m:.3f}", ha="center", fontsize=10)
ax.set_ylim(0.4, 0.85)
ax.set_ylabel("Test accuracy")
ax.set_title("Multi-seed accuracy (5 seeds, mean ± std)")
ax.axhline(0.678, color="gray", linestyle="--", alpha=0.5, linewidth=1)
plt.tight_layout()
plt.savefig(out / "fig_multiseed.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved fig_reliability.png and fig_multiseed.png to iclr_results/figures/")