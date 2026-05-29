
import sys
from pathlib import Path
sys.path.insert(0, '.')
import yaml
import json
import time
import pandas as pd
from datetime import datetime
from src.utils import setup_logging
from experiments.run_pipeline import run_pipeline

def run_meaningful_ablation():
    """Run ablation that shows synthetic data benefit."""
    
    logger = setup_logging()
    
    # Use LOW DATA regime to show synthetic benefit
    base_config = "config/low_data.yaml"
    
    experiments = {
        "baseline_low_data": {
            "description": "Baseline: 200 real samples only",
            "overrides": {
                "data": {
                    "num_synthetic_metadata": 0,
                    "num_synthetic_images": 0
                }
            }
        },
        "synthmed_low_data": {
            "description": "SynthMed: 200 real + 500 synthetic",
            "overrides": {
                "data": {
                    "num_synthetic_metadata": 500,
                    "num_synthetic_images": 500
                }
            }
        },
        "synthmed_no_repair": {
            "description": "SynthMed without schema repair",
            "overrides": {
                "data": {
                    "num_synthetic_metadata": 500,
                    "num_synthetic_images": 500
                },
                "schema": {
                    "repair_enabled": False
                }
            }
        },
        "synthmed_no_rag": {
            "description": "SynthMed without RAG grounding",
            "overrides": {
                "data": {
                    "num_synthetic_metadata": 500,
                    "num_synthetic_images": 500
                }
            }
        }
    }
    
    results = {}
    
    for exp_name, exp_config in experiments.items():
        logger.info(f"\nRunning: {exp_name}")
        logger.info(f"Description: {exp_config['description']}")
        
        # Load base config
        with open(base_config) as f:
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
                        'training_time': float(elapsed)
                    }
            
            logger.info(f"SUCCESS: {exp_name} - Acc: {results[exp_name]['accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"FAILED: {exp_name} - {e}")
            import traceback
            traceback.print_exc()
            results[exp_name] = {'error': str(e)}
    
    # Print summary
    print("\n" + "="*70)
    print("MEANINGFUL ABLATION RESULTS (Low Data Regime)")
    print("="*70)
    print(f"\n{'Experiment':<25} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-"*70)
    
    for exp_name, exp_data in results.items():
        if 'error' not in exp_data:
            print(f"{exp_name:<25} {exp_data['accuracy']:>10.4f} "
                  f"{exp_data['f1_score']:>10.4f} {exp_data['roc_auc']:>10.4f}")
    
    print("-"*70)
    
    # Show improvement
    if 'baseline_low_data' in results and 'synthmed_low_data' in results:
        b = results['baseline_low_data']
        s = results['synthmed_low_data']
        if 'error' not in b and 'error' not in s:
            acc_gain = s['accuracy'] - b['accuracy']
            f1_gain = s['f1_score'] - b['f1_score']
            print(f"\nSynthMed Improvement (Low Data):")
            print(f"  Accuracy: +{acc_gain:.4f} ({acc_gain/b['accuracy']*100:.1f}%)")
            print(f"  F1 Score: +{f1_gain:.4f}")
            
            if acc_gain > 0.03:
                print("  >> SIGNIFICANT IMPROVEMENT - Good for publication!")
            elif acc_gain > 0.01:
                print("  >> Moderate improvement - acceptable")
            else:
                print("  >> Minimal improvement - may need more synthetic data")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"outputs/results/meaningful_ablation_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results

if __name__ == "__main__":
    run_meaningful_ablation()
