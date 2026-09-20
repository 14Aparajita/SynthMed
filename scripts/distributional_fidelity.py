"""
Quantify the validity-utility gap: compare marginal and joint distributions of
real clinical metadata vs. SynthMed-generated metadata.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, spearmanr

FINDING_MAP = {"none": 0, "few": 1, "moderate": 2, "many": 3}

def load_real():
    df = pd.read_csv("data/raw/clinical.csv")
    # Use only the training split of the real low-resource pool
    if "split" in df.columns:
        df = df[df["split"] == "train"]
    # Flatten nested anatomical_findings if present, else use existing columns
    for finding in ["microaneurysms", "hemorrhages", "exudates"]:
        if finding not in df.columns and f"finding_{finding}" in df.columns:
            df[finding] = df[f"finding_{finding}"]
    return df.head(100).reset_index(drop=True)

def load_synth():
    rows = []
    with open("outputs/results/synthetic_metadata_A2.jsonl") as f:
        for line in f:
            r = json.loads(line)
            anat = r.get("anatomical_findings", {})
            rows.append({
                "age": r.get("age", np.nan),
                "image_quality": r.get("image_quality", np.nan),
                "dr_grade": r.get("dr_grade", r.get("_assigned_grade", np.nan)),
                "microaneurysms": anat.get("microaneurysms", "none"),
                "hemorrhages":    anat.get("hemorrhages", "none"),
                "exudates":       anat.get("exudates", "none"),
            })
    return pd.DataFrame(rows)

def js_divergence(a, b, bins=10):
    a = a.dropna().astype(float)
    b = b.dropna().astype(float)
    edges = np.histogram_bin_edges(pd.concat([a, b]), bins=bins)
    p, _ = np.histogram(a, bins=edges, density=True)
    q, _ = np.histogram(b, bins=edges, density=True)
    p = p + 1e-10
    q = q + 1e-10
    return float(jensenshannon(p, q))

def chi2_joint(real, synth, finding):
    r = pd.crosstab(real["dr_grade"], real[finding])
    s = pd.crosstab(synth["dr_grade"], synth[finding])
    cols = sorted(set(r.columns) | set(s.columns))
    r = r.reindex(columns=cols, fill_value=0)
    s = s.reindex(columns=cols, fill_value=0)
    table = np.vstack([r.values, s.values])
    chi2, p, dof, _ = chi2_contingency(table)
    return float(chi2), float(p)

def main():
    real = load_real()
    synth = load_synth()
    print(f"Real records: {len(real)}, Synthetic records: {len(synth)}")

    rows = []

    # Marginal JS divergences
    for field in ["age", "image_quality"]:
        if field not in real.columns:
            continue
        js = js_divergence(real[field], synth[field])
        rows.append({"analysis": "marginal_JS", "field": field,
                     "value": round(js, 4)})
        print(f"JS({field}) = {js:.4f}")

    # Joint chi-square and correlation
    for finding in ["microaneurysms", "hemorrhages", "exudates"]:
        if finding not in real.columns:
            continue
        chi2, p = chi2_joint(real, synth, finding)
        real_corr = spearmanr(
            real["dr_grade"], real[finding].map(FINDING_MAP)
        )[0]
        synth_corr = spearmanr(
            synth["dr_grade"], synth[finding].map(FINDING_MAP)
        )[0]
        rows.append({"analysis": "joint_chi2", "field": finding,
                     "value": round(chi2, 3), "p": round(p, 6)})
        rows.append({"analysis": "corr_real", "field": finding,
                     "value": round(float(real_corr), 3)})
        rows.append({"analysis": "corr_synth", "field": finding,
                     "value": round(float(synth_corr), 3)})
        print(f"{finding}: chi2={chi2:.2f} p={p:.4f} "
              f"real_rho={real_corr:.3f} synth_rho={synth_corr:.3f}")

    out = Path("iclr_results/metrics")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "distributional_fidelity.csv", index=False)
    print(f"\nSaved to {out / 'distributional_fidelity.csv'}")

if __name__ == "__main__":
    main()