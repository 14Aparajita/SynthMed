# Fix build_final_table.py to include the FID column and rerun

$ErrorActionPreference = "Continue"

$code = @'
"""
Assemble the final paper table from real result files.
"""

import json
from pathlib import Path
import pandas as pd

OUT = Path("iclr_results/tables")
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = Path("outputs/results")


def load_metrics(name):
    p = RESULTS / f"{name}_metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_multiseed():
    p = RESULTS / "multiseed_summary_128.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out = {}
    for _, row in df.iterrows():
        out[str(row["experiment"])] = {
            "accuracy_mean": float(row["accuracy_mean"]),
            "accuracy_std": float(row["accuracy_std"]),
            "f1_mean": float(row["f1_mean"]),
            "f1_std": float(row["f1_std"]),
            "roc_auc_mean": float(row["roc_auc_mean"]),
            "roc_auc_std": float(row["roc_auc_std"]),
        }
    return out


def load_fid():
    p = RESULTS / "fid.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f).get("fid", None)


def fmt(v, d=4):
    if v is None or v == -1.0:
        return "-"
    try:
        return f"{float(v):.{d}f}"
    except (ValueError, TypeError):
        return "-"


def main():
    multi = load_multiseed()
    fid_value = load_fid()
    print(f"Multiseed entries: {list(multi.keys())}")
    print(f"FID: {fid_value}")

    rows_spec = [
        ("A1_baseline_100real",          "A1 (real only, 100)",          100, 0,   "multi", "baseline"),
        ("A2_synthmed_100real_500syn",   "A2 (SynthMed, 100 + 500)",     100, 500, "multi", "headline"),
        ("A8_geometric_only",            "A8 (heavy geo aug, 100)",      100, 0,   "multi", "geometric baseline"),
        ("A2_fusion_metadataclassifier", "A2 + metadata fusion",         100, 500, "json",  "fusion ablation"),
        ("A2_hightemp_repair",           "A2 (temperature 1.2)",         100, 500, "json",  "repair ablation"),
    ]

    rows = []
    for name, label, n_real, n_synth, source, note in rows_spec:
        m = load_metrics(name)
        if source == "multi" and name in multi:
            s = multi[name]
            acc = f"{s['accuracy_mean']:.4f} +/- {s['accuracy_std']:.4f}"
            f1  = f"{s['f1_mean']:.4f} +/- {s['f1_std']:.4f}"
            auc = f"{s['roc_auc_mean']:.4f} +/- {s['roc_auc_std']:.4f}"
        elif m is not None:
            acc = fmt(m.get("accuracy"))
            f1  = fmt(m.get("f1_score"))
            auc = fmt(m.get("roc_auc"))
        else:
            continue

        rows.append({
            "Configuration": label,
            "Real": n_real,
            "Synth": n_synth,
            "Accuracy": acc,
            "F1": f1,
            "ROC-AUC": auc,
            "FID": fmt(fid_value, 2) if fid_value and fid_value > 0 and n_synth > 0 else "-",
            "Schema valid": fmt(m.get("schema_validity_rate")) if m else "-",
            "Grounding": fmt(m.get("mean_grounding_score"), 4) if m else "-",
            "Notes": note,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "main_results.csv", index=False)
    with open(OUT / "main_results.md", "w") as f:
        f.write(df.to_markdown(index=False))
    print()
    print(df.to_string(index=False))
    print()
    print(f"Saved: {OUT / 'main_results.csv'}")


if __name__ == "__main__":
    main()
'@

$code | Out-File -FilePath "scripts/build_final_table.py" -Encoding utf8
Write-Host "Wrote scripts/build_final_table.py"

Write-Host ""
python scripts/build_final_table.py