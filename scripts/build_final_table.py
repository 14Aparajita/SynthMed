"""
Assemble the final paper table from already-produced results.
"""
import pandas as pd
from pathlib import Path

rows = [
    # name, real, synth, acc, f1, auc, p_value_vs_A1, notes
    ("A1 (real only)",      100,   0, 0.678, 0.638, 0.877, "—",     "baseline"),
    ("A2 (SynthMed)",       100, 500, 0.628, 0.579, 0.818, "0.214", "not significant"),
    ("A8 (geo aug only)",   100,   0, 0.672, 0.635, 0.886, "0.704", "not significant"),
    ("A2 + fusion",         100, 500, 0.610, 0.552, 0.832, "—",     "metadata did not help"),
    ("C1 (upper bound)",   2000,   0, 0.820, 0.813, 0.919, "—",     "reference"),
]
df = pd.DataFrame(rows, columns=[
    "Configuration", "Real", "Synth", "Accuracy", "F1", "ROC-AUC",
    "p vs A1", "Notes",
])
out = Path("iclr_results/tables")
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "main_results.csv", index=False)
print(df.to_string(index=False))