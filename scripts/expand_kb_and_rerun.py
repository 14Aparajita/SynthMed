"""
Rebuild the knowledge base with 60 documents, then rerun A2 to update
the grounding score under the extended corpus.
"""

import os
os.environ["SYNTHMED_SKIP_FID"] = "1"

import sys
import json
import shutil
from pathlib import Path
sys.path.insert(0, ".")

# 1. Build extended KB
from scripts.build_extended_kb import main as build_kb
build_kb()

# 2. Point the A2 config at the extended KB
cfg_path = Path("config/exp_A2_synthmed_100real_500syn.yaml")
with open(cfg_path) as f:
    cfg = yaml.safe_load(f) if False else None  # placeholder

import yaml
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
cfg["data"]["knowledge_base_dir"] = "data/knowledge_base"  # script auto-loads all .jsonl
with open(cfg_path, "w") as f:
    yaml.dump(cfg, f)

# 3. Rerun A2
from experiments.run_pipeline import run_pipeline
run_pipeline(str(cfg_path))
print("Extended-KB A2 rerun complete. Check outputs/results/A2_..._metrics.json for new grounding score.")