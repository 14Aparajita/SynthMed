"""
Run ablation experiments comparing different SynthMed configurations.
"""

import sys
import yaml
from pathlib import Path
import subprocess
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging

logger = setup_logging()

def run_ablation(ablation_config_path: str = "config/experiment_ablation.yaml"):
    """Run all ablation experiments."""
    with open(ablation_config_path, 'r') as f:
        ablation = yaml.safe_load(f)
    
    logger.info(f"Running {len(ablation['experiments'])} ablation experiments")
    
    for exp in ablation['experiments']:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment: {exp['name']}")
        logger.info(f"{'='*60}")
        
        # Build command
        cmd = [
            sys.executable,
            "experiments/run_pipeline.py",
            "--config", "config/default.yaml",
            f"--experiment.name={exp['name']}",
            f"--data.num_synthetic_metadata={exp['synthetic_metadata']}",
            f"--data.num_synthetic_images={exp['synthetic_images']}",
            f"--schema.repair_enabled={str(exp['schema_repair']).lower()}",
        ]
        
        # Run experiment
        subprocess.run(cmd, check=True)
    
    logger.info("\n" + "="*60)
    logger.info("All ablation experiments complete!")
    logger.info("Results saved in outputs/results/")
    logger.info("="*60)

if __name__ == "__main__":
    run_ablation()