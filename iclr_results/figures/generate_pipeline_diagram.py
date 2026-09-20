"""
Generate a publication-quality pipeline diagram for SynthMed.
Output: iclr_results/figures/fig_pipeline.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---- Style ---------------------------------------------------------------
plt.rcParams.update({
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.6,
})

BG_INPUT   = "#E8F0FE"
BG_PROC    = "#FFF4E5"
BG_RAG     = "#E6F4EA"
BG_META    = "#FCE4EC"
BG_IMG     = "#EDE7F6"
BG_CLS     = "#E0F7FA"
BG_EVAL    = "#F3E5F5"
EDGE       = "#333333"

def box(ax, x, y, w, h, text, facecolor, fontsize=8.5, bold=False):
    """Draw a rounded box with centered text."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=0.8, edgecolor=EDGE, facecolor=facecolor, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", zorder=3,
        fontsize=fontsize, fontweight="bold" if bold else "normal",
        wrap=True,
    )

def arrow(ax, x1, y1, x2, y2, style="-|>", color=EDGE, lw=1.0, ls="-"):
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=10,
        linewidth=lw, color=color, linestyle=ls, zorder=1,
        shrinkA=0, shrinkB=0,
    )
    ax.add_patch(a)

# ---- Figure --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 6.6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")

# ============ LEFT COLUMN: DATA / PIPELINE ============================
# Row 1 — Inputs
box(ax, 0.2, 6.1, 2.4, 0.7, "APTOS 2019\n(3,662 fundus images, 5 DR grades)", BG_INPUT, 8)
box(ax, 0.2, 4.9, 2.4, 0.7, "Clinical metadata\n(clinical.csv)", BG_INPUT, 8)
box(ax, 0.2, 3.7, 2.4, 0.7, "Clinical knowledge base\n(10 DR documents)", BG_RAG, 8)

# Row 2 — Preprocessing + split
box(ax, 3.0, 6.1, 2.2, 0.7, "Preprocess\n128×128, CLAHE, .npy", BG_PROC, 8)
box(ax, 3.0, 4.9, 2.2, 0.7, "Fixed stratified split\n100 train / 100 test", BG_PROC, 8)

# Row 3 — RAG
box(ax, 3.0, 3.7, 2.2, 0.7, "MiniLM-L6-v2\n+ FAISS + RAG Fusion", BG_RAG, 8)

# Row 4 — Metadata generation
box(ax, 5.6, 5.5, 2.4, 0.85,
    "DistilGPT-2\nMetadata generation\n(grade prompt + context)", BG_META, 8)
box(ax, 5.6, 4.4, 2.4, 0.85,
    "JSON Schema validation\n+ bounded rule repair\n(≤3 iterations)", BG_META, 8)

# Row 5 — Image generation (parallel branch)
box(ax, 5.6, 2.8, 2.4, 0.85,
    "DDPM (5M params)\n20 epochs, 32×32 → 128×128\n500 synthetic images", BG_IMG, 8)

# Row 6 — Pairing / dataset
box(ax, 8.4, 4.9, 3.2, 0.9,
    "SyntheticDataset\n(paired image + metadata vector)", BG_PROC, 8)

# Row 7 — Real data pool feeds into pairing
box(ax, 8.4, 3.7, 3.2, 0.85,
    "Real training set (100 images)\n+ synthetic (500 images)", BG_PROC, 8)

# Row 8 — Classifier
box(ax, 8.4, 2.6, 3.2, 0.75,
    "MobileNetV2 classifier\nAdamW, mixup, early stopping", BG_CLS, 8)

# Row 9 — Evaluation
box(ax, 5.6, 1.4, 3.2, 0.75,
    "Evaluate on fixed 100-image test set\n(accuracy, F1, ROC-AUC)", BG_EVAL, 8)

box(ax, 9.4, 1.4, 2.2, 0.75,
    "Multi-seed\n(5 seeds: 42–46)", BG_EVAL, 8)

# Row 10 — Reliability outputs
box(ax, 0.2, 1.4, 4.6, 0.9,
    "Reliability outputs:\nschema validity = 1.000 (n=500)  |  repair success = 1.000 (n=100)",
    BG_META, 8.5, bold=True)

# ============ ARROWS ====================================================
# Inputs → Preprocess
arrow(ax, 2.6, 6.45, 3.0, 6.45)
arrow(ax, 2.6, 5.25, 3.0, 5.25)
arrow(ax, 2.6, 4.05, 3.0, 4.05)

# Preprocess → split (vertical)
arrow(ax, 4.1, 6.1, 4.1, 5.6)
arrow(ax, 4.1, 4.9, 4.1, 4.4)

# Split → metadata gen
arrow(ax, 5.2, 5.25, 5.6, 5.7)

# RAG → metadata gen
arrow(ax, 5.2, 4.05, 5.6, 5.6, style="-|>", color="#1B5E20", lw=1.2, ls="--")

# Metadata gen → validation
arrow(ax, 6.8, 5.5, 6.8, 5.25)

# Split → DDPM
arrow(ax, 5.2, 5.25, 6.8, 3.65, style="-|>", color=EDGE, lw=1.0)

# Validation → pairing
arrow(ax, 8.0, 4.82, 8.4, 5.25)

# DDPM → pairing
arrow(ax, 8.0, 3.22, 8.4, 4.9)

# Real pool → pairing
arrow(ax, 10.0, 4.55, 10.0, 4.55)  # no-op decorative
arrow(ax, 10.0, 3.7, 10.0, 4.9, style="-|>")

# Pairing → classifier
arrow(ax, 10.0, 4.9, 10.0, 3.35)

# Classifier → eval
arrow(ax, 9.0, 2.6, 7.5, 2.15)

# Eval → multiseed
arrow(ax, 8.8, 1.78, 9.4, 1.78)

# Reliability outputs stand alone (feed from validation path)
arrow(ax, 5.6, 4.82, 2.8, 2.3, style="-|>", color="#880E4F", lw=1.1, ls=":")

# ============ LEGEND ====================================================
legend_items = [
    ("Inputs",             BG_INPUT),
    ("Preprocessing",      BG_PROC),
    ("RAG grounding",      BG_RAG),
    ("Metadata pipeline",  BG_META),
    ("Image generation",   BG_IMG),
    ("Classifier",         BG_CLS),
    ("Evaluation",         BG_EVAL),
]
lx, ly = 0.2, 0.35
for i, (label, color) in enumerate(legend_items):
    x = lx + i * 1.55
    patch = FancyBboxPatch((x, ly), 0.25, 0.25,
                           boxstyle="round,pad=0.01",
                           linewidth=0.5, edgecolor=EDGE, facecolor=color)
    ax.add_patch(patch)
    ax.text(x + 0.3, ly + 0.125, label, fontsize=7.5, va="center")

# Title
ax.text(6, 6.95, "SynthMed Pipeline",
        ha="center", va="center", fontsize=13, fontweight="bold")

plt.tight_layout()
out = Path("iclr_results/figures/fig_pipeline.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved pipeline diagram to {out}")