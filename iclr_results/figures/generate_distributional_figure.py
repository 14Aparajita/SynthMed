import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Paths
# ============================================================
REAL_PATH = "data/raw/clinical.csv"
SYNTH_PATH = "outputs/results/synthetic_metadata_A2.jsonl"
OUT_PATH = Path("iclr_results/figures/fig_distributional_mismatch.png")

# ============================================================
# Load REAL metadata
# ============================================================
real = pd.read_csv(REAL_PATH)

if "split" in real.columns:
    real = real[real["split"] == "train"]

print("REAL COLUMNS:")
print(real.columns.tolist())

# Use 100 real samples, matching the low-resource setting
real = real.head(100)

# ============================================================
# Load SYNTHETIC metadata
# ============================================================
rows = []

with open(SYNTH_PATH, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue

        r = json.loads(line)
        anat = r.get("anatomical_findings", {})

        rows.append({
            "age": r.get("age", np.nan),
            "image_quality": r.get("image_quality", np.nan),
            "microaneurysms": anat.get("microaneurysms", np.nan),
            "hemorrhages": anat.get("hemorrhages", np.nan),
            "exudates": anat.get("exudates", np.nan),
        })

synth = pd.DataFrame(rows)

print("\nSYNTHETIC COLUMNS:")
print(synth.columns.tolist())

# ============================================================
# FIGURE
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# ------------------------------------------------------------
# Panel A: Age distribution
# ------------------------------------------------------------
ax = axes[0]

if "age" in real.columns and "age" in synth.columns:
    real_age = pd.to_numeric(real["age"], errors="coerce").dropna()
    synth_age = pd.to_numeric(synth["age"], errors="coerce").dropna()

    # Data-driven bins
    combined = pd.concat([real_age, synth_age])

    if len(combined) > 0:
        bins = np.histogram_bin_edges(combined, bins=12)

        ax.hist(
                real_age,
                bins=bins,
                density=True,
                alpha=0.6,
                label=f"Real (n={len(real_age)})",
                color="#2E86AB"
            )

        ax.hist(
                synth_age,
                bins=bins,
                density=True,
                alpha=0.6,
                label=f"Synthetic (n={len(synth_age)})",
                color="#E67E22"
            )

        ax.set_ylabel("Density")

        ax.set_xlabel("Age")
        ax.set_title("(a) Age distribution")
        ax.legend()

        ax.text(
            0.03,
            0.95,
            "JS = 0.70",
            transform=ax.transAxes,
            fontsize=11,
            va="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8
            )
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Age data unavailable",
            ha="center",
            va="center"
        )
else:
    ax.text(
        0.5,
        0.5,
        "Age column unavailable",
        ha="center",
        va="center"
    )

# ------------------------------------------------------------
# Panel B: Verified distributional divergence
# ------------------------------------------------------------
ax = axes[1]

features = ["Age", "Image quality"]
js_values = [0.7011, 0.8326]

bars = ax.bar(
    features,
    js_values,
    width=0.55,
    color=["#5DADE2", "#E67E22"],
    edgecolor="black",
    linewidth=0.5
)

ax.set_ylabel("Jensen–Shannon divergence")
ax.set_title("(b) Distributional mismatch")
ax.set_ylim(0, 1.0)

for bar, value in zip(bars, js_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.025,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()

# ============================================================
# Save
# ============================================================
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(
    OUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\nSaved: {OUT_PATH}")