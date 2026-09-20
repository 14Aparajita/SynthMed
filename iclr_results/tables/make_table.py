import pandas as pd
import os

os.makedirs("iclr_results/tables", exist_ok=True)
df = pd.read_csv("iclr_results/metrics/final_results.csv")
table = df[["experiment", "accuracy", "f1_score", "roc_auc"]].copy()
table.columns = ["Experiment", "Accuracy", "F1", "ROC-AUC"]
table.to_csv("iclr_results/tables/results_summary.csv", index=False)

# Markdown version
with open("iclr_results/tables/results_summary.md", "w") as f:
    f.write("| Experiment | Accuracy | F1 | ROC-AUC |\n")
    f.write("|------------|----------|-----|---------|\n")
    for _, row in table.iterrows():
        f.write(f"| {row['Experiment']} | {row['Accuracy']:.4f} | {row['F1']:.4f} | {row['ROC-AUC']:.4f} |\n")
print("Saved results table")