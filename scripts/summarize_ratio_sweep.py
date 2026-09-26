import json
from pathlib import Path
import pandas as pd

RESULTS = Path("outputs/results")
ratios = [0, 200, 500, 1000]
rows = []

for n in ratios:
    path = RESULTS / f"ratio_{n}_metrics.json"
    if not path.exists():
        print(f"MISSING: {path}")
        continue
    with open(path) as f:
        m = json.load(f)
    rows.append({
        "n_synth": n,
        "accuracy": m.get("accuracy", 0.0),
        "f1_score": m.get("f1_score", 0.0),
        "roc_auc": m.get("roc_auc", 0.0),
        "schema_validity": m.get("schema_validity_rate", 0.0),
        "repair_success": m.get("repair_success_rate", 0.0),
        "grounding": m.get("mean_grounding_score", 0.0),
    })

if rows:
    df = pd.DataFrame(rows)
    out = RESULTS / "ratio_sweep.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {out}")
else:
    print("No ratio metrics files found.")
