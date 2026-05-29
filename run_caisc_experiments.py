# save as: run_caisc_experiments.py
# ACTUALLY runs the experiments for CAISC 2026 paper

import sys
from pathlib import Path
sys.path.insert(0, '.')
import yaml
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

from src.utils import setup_logging, set_seed
from experiments.run_pipeline import run_pipeline

logger = setup_logging()
set_seed(42)

print("="*70)
print("CAISC 2026 - REAL EXPERIMENTAL RUNS")
print("="*70)

# Load and shuffle data
df = pd.read_csv("data/raw/clinical.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Define the key experiments for the paper
# We focus on low-data regime where synthetic data matters most
experiments = [
    {
        "name": "A1_baseline_100real",
        "description": "Baseline: 100 real samples only",
        "n_real": 100,
        "n_synthetic_meta": 0,
        "n_synthetic_img": 0,
        "repair": False,
        "rag": False
    },
    {
        "name": "A2_synthmed_100real_500syn",
        "description": "SynthMed: 100 real + 500 synthetic",
        "n_real": 100,
        "n_synthetic_meta": 500,
        "n_synthetic_img": 500,
        "repair": True,
        "rag": True
    },
    {
        "name": "A3_norepair_100real_500syn",
        "description": "Ablation: No schema repair",
        "n_real": 100,
        "n_synthetic_meta": 500,
        "n_synthetic_img": 500,
        "repair": False,
        "rag": True
    },
    {
        "name": "A4_norag_100real_500syn",
        "description": "Ablation: No RAG grounding",
        "n_real": 100,
        "n_synthetic_meta": 500,
        "n_synthetic_img": 500,
        "repair": True,
        "rag": False
    },
    {
        "name": "B1_baseline_200real",
        "description": "Baseline: 200 real samples only",
        "n_real": 200,
        "n_synthetic_meta": 0,
        "n_synthetic_img": 0,
        "repair": False,
        "rag": False
    },
    {
        "name": "B2_synthmed_200real_500syn",
        "description": "SynthMed: 200 real + 500 synthetic",
        "n_real": 200,
        "n_synthetic_meta": 500,
        "n_synthetic_img": 500,
        "repair": True,
        "rag": True
    },
    {
        "name": "C1_upperbound_fulldata",
        "description": "Upper bound: All real data (2564 samples)",
        "n_real": 2000,
        "n_synthetic_meta": 0,
        "n_synthetic_img": 0,
        "repair": False,
        "rag": False
    },
]

results = []

for exp in experiments:
    logger.info(f"\n{'='*70}")
    logger.info(f"RUNNING: {exp['name']}")
    logger.info(f"Description: {exp['description']}")
    logger.info(f"{'='*70}")
    
    # Create stratified sample of real data
    n_real = min(exp['n_real'], len(df))
    train_sample, remaining = train_test_split(
        df, 
        train_size=n_real,
        stratify=df['dr_grade'],
        random_state=42
    )
    
    # Save temporary clinical.csv with only the sampled data
    temp_clinical = train_sample.copy()
    temp_clinical['split'] = 'train'
    
    # Add test data from remaining
    n_test = min(100, len(remaining))
    test_sample = remaining.iloc[:n_test].copy()
    test_sample['split'] = 'test'
    
    temp_df = pd.concat([temp_clinical, test_sample])
    temp_df.to_csv("data/raw/clinical_temp.csv", index=False)
    
    # Backup original
    import shutil
    shutil.copy("data/raw/clinical.csv", "data/raw/clinical_backup.csv")
    temp_df.to_csv("data/raw/clinical.csv", index=False)
    
    # Create experiment config
    config_dict = {
        'experiment': {
            'name': exp['name'],
            'seed': 42,
            'device': 'cuda'
        },
        'data': {
            'raw_dir': 'data/raw',
            'processed_dir': 'data/processed',
            'knowledge_base_dir': 'data/knowledge_base',
            'image_size': 128,
            'num_real_train': n_real,
            'num_real_test': n_test,
            'num_synthetic_metadata': exp['n_synthetic_meta'],
            'num_synthetic_images': exp['n_synthetic_img'],
        },
        'schema': {
            'schema_path': 'config/schema/clinical_metadata.json',
            'repair_enabled': exp['repair'],
            'repair_max_iterations': 3,
        },
        'retrieval': {
            'embedder_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'top_k': 5,
            'fusion_weights': [0.4, 0.3, 0.3],
            'index_path': 'outputs/models/faiss_index.bin',
        },
        'generation': {
            'metadata_model': 'distilgpt2',
            'metadata_max_length': 256,
            'temperature': 0.7,
            'diffusion_timesteps': 100,
            'diffusion_image_size': 32,
            'diffusion_checkpoint': 'outputs/models/diffusion_unet.pt',
        },
        'classifier': {
            'model_name': 'mobilenet_v2',
            'num_classes': 5,
            'batch_size': 8,
            'epochs': 50,
            'learning_rate': 0.0001,
            'weight_decay': 0.001,
        },
        'evaluation': {
            'metrics': ['accuracy', 'f1', 'roc_auc'],
            'save_results': True,
            'results_path': 'outputs/results',
        }
    }
    
    # Save config
    config_path = f"config/exp_{exp['name']}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    # Run the actual pipeline
    start_time = time.time()
    try:
        run_pipeline(config_path)
        elapsed = time.time() - start_time
        
        # Read the actual results
        results_df = pd.read_csv("outputs/results/experiment_comparison.csv")
        if len(results_df) > 0:
            row = results_df.iloc[-1]
            
            result = {
                'experiment': exp['name'],
                'description': exp['description'],
                'n_real': n_real,
                'n_synthetic': exp['n_synthetic_meta'],
                'accuracy': float(row['accuracy']),
                'f1_score': float(row['f1_score']),
                'roc_auc': float(row['roc_auc']),
                'repair_enabled': exp['repair'],
                'rag_enabled': exp['rag'],
                'training_time': elapsed,
            }
            results.append(result)
            
            logger.info(f"COMPLETED: {exp['name']}")
            logger.info(f"  Accuracy: {result['accuracy']:.4f}")
            logger.info(f"  F1: {result['f1_score']:.4f}")
            logger.info(f"  ROC-AUC: {result['roc_auc']:.4f}")
            logger.info(f"  Time: {elapsed:.0f}s")
        
    except Exception as e:
        logger.error(f"FAILED: {exp['name']} - {e}")
        import traceback
        traceback.print_exc()
    
    # Restore original clinical.csv
    shutil.copy("data/raw/clinical_backup.csv", "data/raw/clinical.csv")

# Restore original
if Path("data/raw/clinical_backup.csv").exists():
    import shutil
    shutil.copy("data/raw/clinical_backup.csv", "data/raw/clinical.csv")

# Print final results table
print("\n\n")
print("="*80)
print("CAISC 2026 - EXPERIMENTAL RESULTS")
print("="*80)
print(f"\n{'Experiment':<35} {'Real':>6} {'Syn':>6} {'Acc':>8} {'F1':>8} {'ROC-AUC':>8}")
print("-"*80)

for r in results:
    print(f"{r['experiment']:<35} {r['n_real']:>6} {r['n_synthetic']:>6} "
          f"{r['accuracy']:>8.4f} {r['f1_score']:>8.4f} {r['roc_auc']:>8.4f}")

print("-"*80)

# Calculate key metrics for paper
if len(results) >= 2:
    baselines = {r['experiment']: r for r in results if 'baseline' in r['experiment'].lower()}
    synthmeds = {r['experiment']: r for r in results if 'synthmed' in r['experiment'].lower()}
    
    print("\nKEY FINDINGS FOR PAPER:")
    for key in baselines:
        if key.replace('baseline', 'synthmed') in synthmeds:
            b = baselines[key]
            s = synthmeds[key.replace('baseline', 'synthmed')]
            acc_gain = s['accuracy'] - b['accuracy']
            f1_gain = s['f1_score'] - b['f1_score']
            print(f"\n{b['description']} vs {s['description']}:")
            print(f"  Accuracy: {b['accuracy']:.4f} -> {s['accuracy']:.4f} (+{acc_gain:.4f})")
            print(f"  F1 Score: {b['f1_score']:.4f} -> {s['f1_score']:.4f} (+{f1_gain:.4f})")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_df = pd.DataFrame(results)
results_df.to_csv(f"outputs/results/caisc_final_results_{timestamp}.csv", index=False)

with open(f"outputs/results/caisc_final_results_{timestamp}.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to: outputs/results/caisc_final_results_{timestamp}.csv")
print("="*80)