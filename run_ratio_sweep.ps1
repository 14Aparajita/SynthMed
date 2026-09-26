# =====================================================================
# run_ratio_sweep.ps1 — runs all four ratios one at a time, unattended
# Each ratio runs as a completely independent python process so the
# Windows paging file is fully reclaimed between runs.
# =====================================================================

$ErrorActionPreference = "Continue"
$log       = "ratio_sweep.log"
$startTime = Get-Date

function Log($msg) {
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Write-Host $line -ForegroundColor Cyan
    Add-Content -Path $log -Value $line
}

Log "=== Ratio sweep starting at $($startTime.ToString('HH:mm:ss')) ==="

# ---------------------------------------------------------------------
# Step A — Ensure summarizer script exists
# ---------------------------------------------------------------------
if (-not (Test-Path "scripts/summarize_ratio_sweep.py")) {
    Log "Creating scripts/summarize_ratio_sweep.py"
    $summarizer = @'
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
'@
    $summarizer | Out-File -FilePath "scripts/summarize_ratio_sweep.py" -Encoding utf8
}

# ---------------------------------------------------------------------
# Step B — Ensure build_final_table.py includes FID
# ---------------------------------------------------------------------
$tableFile = "scripts/build_final_table.py"
if (Test-Path $tableFile) {
    $content = Get-Content $tableFile -Raw
    if ($content -notmatch 'fid_path = RESULTS / "fid.json"') {
        Log "Patching build_final_table.py to include FID"
        $content = $content -replace '(multi = load_multiseed\(\))', @"
`$1

    fid_path = RESULTS / "fid.json"
    fid_value = None
    if fid_path.exists():
        with open(fid_path) as f:
            fid_value = json.load(f).get("fid", None)
"@
        $content = $content -replace '"FID": fmt\(m\.get\("fid"\)\) if m else "-"', '"FID": fmt(fid_value) if fid_value and fid_value > 0 else (fmt(m.get("fid")) if m else "-")'
        $content | Out-File -FilePath $tableFile -Encoding utf8
    }
}

# ---------------------------------------------------------------------
# Step C — Environment
# ---------------------------------------------------------------------
$env:SYNTHMED_SKIP_FID = "1"
New-Item -ItemType Directory -Force -Path "outputs/logs" | Out-Null

# ---------------------------------------------------------------------
# Step D — Run each ratio independently
# ---------------------------------------------------------------------
$ratios = @(0, 200, 500, 1000)

foreach ($n in $ratios) {
    $cfg     = "config/exp_ratio_$n.yaml"
    $metrics = "outputs/results/ratio_${n}_metrics.json"
    $stdout  = "outputs/logs/ratio_${n}_stdout.log"
    $stderr  = "outputs/logs/ratio_${n}_stderr.log"

    Log ""
    Log "----------------------------------------------------"
    Log "Ratio $n"
    Log "----------------------------------------------------"

    if (-not (Test-Path $cfg)) {
        Log "  MISSING config: $cfg -- skipping"
        continue
    }

    # Skip if metrics file was written in the last 4 hours
    if (Test-Path $metrics) {
        $age = (Get-Date) - (Get-Item $metrics).LastWriteTime
        if ($age.TotalMinutes -lt 240) {
            Log "  Already completed $([math]::Round($age.TotalMinutes,0)) min ago -- skipping"
            continue
        }
    }

    Log "  Spawning python subprocess..."
    $proc = Start-Process -FilePath "python" `
        -ArgumentList "experiments/run_pipeline.py", "--config", $cfg `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError  $stderr

    if ($proc.ExitCode -eq 0 -and (Test-Path $metrics)) {
        Log "  SUCCESS"
    } else {
        Log "  FAILED (exit=$($proc.ExitCode)) -- see $stderr"
    }

    # Force OS to reclaim memory
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
    Start-Sleep -Seconds 20
    Log "  Cooled down 20s"
}

# ---------------------------------------------------------------------
# Step E — Aggregate + rebuild table
# ---------------------------------------------------------------------
Log ""
Log "=== Aggregating results ==="
python scripts/summarize_ratio_sweep.py

Log ""
Log "=== Rebuilding paper table ==="
python scripts/build_final_table.py

$elapsed = (Get-Date) - $startTime
Log ""
Log "=== DONE. Total elapsed: $([math]::Round($elapsed.TotalMinutes,1)) min ==="
Log "Log file: $log"