# save as: generate_paper_figures.py
# Creates publication-quality figures for your paper

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Your actual results
data = {
    'Experiment': ['100 Real\n(Baseline)', '100 Real + 500 Syn\n(SynthMed)', 
                   '200 Real\n(Baseline)', '2000 Real\n(Upper Bound)'],
    'Accuracy': [0.680, 0.740, 0.720, 0.850],
    'F1': [0.620, 0.717, 0.677, 0.846],
    'ROC_AUC': [0.903, 0.927, 0.913, 0.954],
    'Color': ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12']
}

# Figure 1: Main Results Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy comparison
ax = axes[0]
bars = ax.bar(range(4), data['Accuracy'], color=data['Color'], edgecolor='black', linewidth=1.5)
ax.set_xticks(range(4))
ax.set_xticklabels(data['Experiment'], fontsize=10)
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('DR Classification Accuracy', fontsize=14, fontweight='bold')
ax.set_ylim(0.5, 0.9)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, acc in zip(bars, data['Accuracy']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{acc:.1%}', ha='center', fontweight='bold', fontsize=11)

# Highlight the improvement
ax.annotate('+6.0%', xy=(1, 0.74), xytext=(1.5, 0.78),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=12, fontweight='bold', color='green')

# F1 comparison
ax = axes[1]
bars = ax.bar(range(4), data['F1'], color=data['Color'], edgecolor='black', linewidth=1.5)
ax.set_xticks(range(4))
ax.set_xticklabels(data['Experiment'], fontsize=10)
ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
ax.set_ylim(0.5, 0.9)
ax.grid(axis='y', alpha=0.3)

for bar, f1 in zip(bars, data['F1']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{f1:.1%}', ha='center', fontweight='bold', fontsize=11)

ax.annotate('+9.7%', xy=(1, 0.717), xytext=(1.5, 0.76),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=12, fontweight='bold', color='green')

plt.tight_layout()
plt.savefig('outputs/results/main_results.png', dpi=300, bbox_inches='tight')
print("Figure 1 saved: main_results.png")

# Figure 2: Data Efficiency Curve
fig, ax = plt.subplots(figsize=(8, 5))

real_samples = [100, 100, 200, 2000]
accuracies = [0.680, 0.740, 0.720, 0.850]
labels = ['Baseline', 'SynthMed', 'Baseline', 'Upper Bound']
colors = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12']
markers = ['o', 's', 'o', 'D']

for i in range(len(real_samples)):
    ax.scatter(real_samples[i], accuracies[i], c=colors[i], s=200, 
              marker=markers[i], edgecolors='black', linewidth=2, zorder=5)
    ax.annotate(labels[i], (real_samples[i], accuracies[i]), 
               textcoords="offset points", xytext=(10, 10), fontsize=10)

# Add connecting lines
ax.plot([100, 100], [0.680, 0.740], 'g--', linewidth=2, alpha=0.7)
ax.plot([100, 2000], [0.680, 0.850], 'r:', linewidth=1.5, alpha=0.5)

ax.set_xlabel('Number of Real Training Samples', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Data Efficiency: SynthMed Bridges the Gap', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.set_ylim(0.6, 0.9)
ax.grid(alpha=0.3)

# Add annotations
ax.annotate('Gap with\n100 real: 17%', xy=(150, 0.71), fontsize=10, color='red')
ax.annotate('Gap with\nSynthMed: 11%', xy=(150, 0.755), fontsize=10, color='green')
ax.annotate('35% gap\nrecovery', xy=(300, 0.73), fontsize=10, color='green',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.savefig('outputs/results/data_efficiency.png', dpi=300, bbox_inches='tight')
print("Figure 2 saved: data_efficiency.png")

# Figure 3: Results Summary Table
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')

table_data = [
    ['100 Real (Baseline)', '68.0%', '62.0%', '90.3%'],
    ['100 Real + 500 Syn (SynthMed)', '74.0% (+6.0%)', '71.7% (+9.7%)', '92.7% (+2.4%)'],
    ['200 Real (Baseline)', '72.0%', '67.7%', '91.3%'],
    ['2000 Real (Upper Bound)', '85.0%', '84.6%', '95.4%'],
]

table = ax.table(cellText=table_data,
                colLabels=['Configuration', 'Accuracy', 'F1 Score', 'ROC-AUC'],
                cellLoc='center',
                loc='center',
                colWidths=[0.35, 0.2, 0.2, 0.2])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Style the header
for i in range(4):
    table[0, i].set_facecolor('#2C3E50')
    table[0, i].set_text_props(color='white', fontweight='bold')

# Highlight SynthMed row
for i in range(4):
    table[1, i].set_facecolor('#D5F5E3')

plt.title('CAISC 2026: SynthMed Experimental Results', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/results/results_table.png', dpi=300, bbox_inches='tight')
print("Figure 3 saved: results_table.png")

print("\nAll figures saved in outputs/results/")
print("Ready for CAISC 2026 paper submission!")