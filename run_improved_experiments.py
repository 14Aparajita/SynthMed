
import sys
from pathlib import Path
sys.path.insert(0, '.')
import yaml
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

from src.utils import setup_logging, set_seed
from experiments.run_pipeline import run_pipeline

def save_config_as_dict(config_path, output_path):
    # Read original config
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Write as plain YAML
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    return output_path

def run_ablation_experiments():
    logger = setup_logging()
    
    experiments = {
        "baseline_real_only": {
            "description": "Baseline: Real data only, no synthetic augmentation",
            "overrides": {
                "data": {
                    "num_synthetic_metadata": 0,
                    "num_synthetic_images": 0
                },
                "schema": {
                    "repair_enabled": False
                }
            }
        },
        "synth_metadata_only": {
            "description": "Synthetic metadata augmentation only",
            "overrides": {
                "data": {
                    "num_synthetic_metadata": 200,
                    "num_synthetic_images": 0
                },
                "schema": {
                    "repair_enabled": True
                }
            }
        },
        "synth_both_full": {
            "description": "Full SynthMed: Both metadata and image generation",
            "overrides": {
                "data": {
                    "num_synthetic_metadata": 200,
                    "num_synthetic_images": 200
                },
                "schema": {
                    "repair_enabled": True
                }
            }
        },
        "no_schema_repair": {
            "description": "Ablation: Without schema repair",
            "overrides": {
                "data": {
                    "num_synthetic_metadata": 200,
                    "num_synthetic_images": 200
                },
                "schema": {
                    "repair_enabled": False
                }
            }
        },
        "no_rag_grounding": {
            "description": "Ablation: Without RAG knowledge grounding",
            "overrides": {
                "data": {
                    "num_synthetic_metadata": 200,
                    "num_synthetic_images": 200
                },
                "schema": {
                    "repair_enabled": True
                }
            }
        }
    }
    
    # Use the improved config as base
    base_config_path = "config/improved.yaml" if Path("config/improved.yaml").exists() else "config/default.yaml"
    
    results = {}
    
    for exp_name, exp_config in experiments.items():
        logger.info("="*70)
        logger.info(f"Running: {exp_name}")
        logger.info(f"Description: {exp_config['description']}")
        logger.info("="*70)
        
        # Load base config as dict
        with open(base_config_path) as f:
            config_dict = yaml.safe_load(f)
        
        # Apply overrides
        for section, values in exp_config['overrides'].items():
            if section in config_dict:
                config_dict[section].update(values)
        
        # Save experiment config
        exp_config_path = f"config/exp_{exp_name}.yaml"
        with open(exp_config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        # Run pipeline
        start_time = time.time()
        try:
            from experiments.run_pipeline import run_pipeline
            run_pipeline(exp_config_path)
            elapsed = time.time() - start_time
            
            # Read results
            results_path = Path("outputs/results/experiment_comparison.csv")
            if results_path.exists():
                results_df = pd.read_csv(results_path)
                if len(results_df) > 0:
                    row = results_df.iloc[-1]
                    results[exp_name] = {
                        'description': exp_config['description'],
                        'accuracy': float(row['accuracy']),
                        'f1_score': float(row['f1_score']),
                        'roc_auc': float(row['roc_auc']),
                        'schema_validity': float(row['schema_validity_rate']),
                        'repair_success': float(row['repair_success_rate']),
                        'grounding_score': float(row['mean_grounding_score']),
                        'training_time': float(elapsed)
                    }
            
            logger.info(f"SUCCESS: {exp_name} completed in {elapsed:.1f}s")
            
        except Exception as e:
            logger.error(f"FAILED: {exp_name} - {e}")
            import traceback
            traceback.print_exc()
            results[exp_name] = {'error': str(e)}
    
    # Print final comparison
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    print(f"\n{'Experiment':<25} {'Acc':>8} {'F1':>8} {'ROC-AUC':>8}")
    print("-"*70)
    
    for exp_name, exp_data in results.items():
        if 'error' not in exp_data:
            print(f"{exp_name:<25} {exp_data['accuracy']:>8.4f} "
                  f"{exp_data['f1_score']:>8.4f} {exp_data['roc_auc']:>8.4f}")
    
    print("-"*70)
    
    # Calculate improvement
    if 'baseline_real_only' in results and 'synth_both_full' in results:
        baseline = results['baseline_real_only']
        synthmed = results['synth_both_full']
        if 'error' not in baseline and 'error' not in synthmed:
            print(f"\nSynthMed Improvement over Baseline:")
            print(f"  Accuracy: {synthmed['accuracy'] - baseline['accuracy']:+.4f}")
            print(f"  F1 Score: {synthmed['f1_score'] - baseline['f1_score']:+.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"outputs/results/final_results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results

if __name__ == "__main__":
    run_ablation_experiments()
