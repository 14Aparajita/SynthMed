# run_caisc_experiments.py (updated)
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
import shutil

from src.utils import setup_logging, set_seed
from experiments.run_pipeline import run_pipeline

logger = setup_logging()
set_seed(42)

print("="*70)
print("CAISC 2026 - REAL EXPERIMENTAL RUNS")
print("="*70)

# Load the FULL original dataset once
full_df = pd.read_csv("data/raw/clinical_full.csv")  # create this file first!
full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save a persistent full copy if not already present
full_df.to_csv("data/raw/clinical_full.csv", index=False)

# Define experiments
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
    {
        "name": "A5_imageonly_diffusion",
        "description": "Image-only diffusion augmentation, no metadata/RAG/repair",
        "n_real": 100,
        "n_synthetic_meta": 0,
        "n_synthetic_img": 500,
        "repair": False,
        "rag": False
    },
    {
        "name": "A8_geometric_only",
        "description": "Geometric augmentation only, no synthetic data",
        "n_real": 100,
        "n_synthetic_meta": 0,
        "n_synthetic_img": 0,
        "repair": False,
        "rag": False
    },
    {
    "name": "A6_synthmed_conditional",
    "description": "SynthMed with class-conditional DDPM",
    "n_real": 100,
    "n_synthetic_meta": 500,
    "n_synthetic_img": 500,
    "repair": True,
    "rag": True,
    "conditioning_enabled": True
},
]

results = []

def get_fixed_test_set(df, n_test=100, seed=42):
    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(df, test_size=n_test, stratify=df['dr_grade'], random_state=seed)
    test_df.to_csv("data/processed/fixed_test_ids.csv", index=False)
    return test_df

fixed_test_df = get_fixed_test_set(full_df, n_test=100, seed=42)
fixed_test_ids = set(fixed_test_df['image_id'])  # ensure column name
pool_df = full_df[~full_df['image_id'].isin(fixed_test_ids)]

for exp in experiments:
    logger.info(f"\n{'='*70}")
    logger.info(f"RUNNING: {exp['name']}")
    logger.info(f"Description: {exp['description']}")
    logger.info(f"{'='*70}")
    
    n_real = min(exp['n_real'], len(pool_df))
    
    # Stratified train sample from pool (excluding fixed test)
    train_sample, _ = train_test_split(
        pool_df,
        train_size=n_real,
        stratify=pool_df['dr_grade'],
        random_state=42
    )
    
    # Create temp clinical.csv with train and fixed test
    temp_df = pd.concat([
        train_sample.assign(split='train'),
        fixed_test_df.assign(split='test')
    ])
    temp_df.to_csv("data/raw/clinical.csv", index=False)
    
    # Backup and replace clinical.csv
    import shutil
    shutil.copy("data/raw/clinical.csv", "data/raw/clinical_backup.csv")
    temp_df.to_csv("data/raw/clinical.csv", index=False)
    
    # Build config dict
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
            'num_real_test': 100,
            'num_synthetic_metadata': exp['n_synthetic_meta'],
            'num_synthetic_images': exp['n_synthetic_img'],
            'augmentation_strength': 'default',
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
            'rag_enabled': exp['rag']
        },
        'generation': {
            'metadata_model': 'distilgpt2',
            'metadata_max_length': 256,
            'temperature': 0.7,
            'diffusion_timesteps': 100,
            'diffusion_image_size': 32,
            'diffusion_checkpoint': 'outputs/models/diffusion_unet.pt',
            'diffusion_epochs': 20,
            'conditioning_enabled': exp.get('conditioning_enabled', False),
        },
        'classifier': {
            'model_name': 'mobilenet_v2',
            'num_classes': 5,
            'batch_size': 8,
            'epochs': 50,
            'learning_rate': 0.0001,
            'weight_decay': 0.001,
            'use_metadata': False,
        },
        'evaluation': {
            'metrics': ['accuracy', 'f1', 'roc_auc'],
            'save_results': True,
            'results_path': 'outputs/results',
        }
    }
    
    config_path = f"config/exp_{exp['name']}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    start_time = time.time()
    try:
        run_pipeline(config_path)
        elapsed = time.time() - start_time
        
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
    
    # Restore original
    shutil.copy("data/raw/clinical_backup.csv", "data/raw/clinical.csv")

# Restore original if backup exists
if Path("data/raw/clinical_backup.csv").exists():
    import shutil
    shutil.copy("data/raw/clinical_backup.csv", "data/raw/clinical.csv")

# Print final results
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

# Save final results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_df = pd.DataFrame(results)
results_df.to_csv(f"outputs/results/caisc_final_results_{timestamp}.csv", index=False)
with open(f"outputs/results/caisc_final_results_{timestamp}.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to: outputs/results/caisc_final_results_{timestamp}.csv")
print("="*80)