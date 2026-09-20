# run_multiseed.py
import sys
from pathlib import Path
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_rel
import yaml
import time
from src.utils import setup_logging, set_seed
from experiments.run_pipeline import run_pipeline

logger = setup_logging()
set_seed(42)

full_df = pd.read_csv("data/raw/clinical_full.csv")
full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

# fixed test set
_, test_df = train_test_split(full_df, test_size=100, stratify=full_df['dr_grade'], random_state=42)
test_ids = set(test_df['image_id'])
pool_df = full_df[~full_df['image_id'].isin(test_ids)]

seeds = [42, 43, 44, 45, 46]
# target_exps = ["A1_baseline_100real", "A2_synthmed_100real_500syn"]
target_exps = ["A1_baseline_100real", "A2_synthmed_100real_500syn", "A8_geometric_only"]

all_results = []

for seed in seeds:
    set_seed(seed)
    for exp_name in target_exps:
        n_real = 100
        train_sample, _ = train_test_split(pool_df, train_size=n_real, stratify=pool_df['dr_grade'], random_state=seed)
        temp_df = pd.concat([train_sample.assign(split='train'), test_df.assign(split='test')])
        temp_df.to_csv("data/raw/clinical.csv", index=False)
        
        # build config dict for this seed/exp
        config_dict = {
            'experiment': {'name': f"{exp_name}_seed{seed}", 'seed': seed, 'device': 'cuda'},
            'data': {'raw_dir': 'data/raw', 'processed_dir': 'data/processed',
                     'knowledge_base_dir': 'data/knowledge_base', 'image_size': 128,
                     'num_real_train': n_real, 'num_real_test': 100,
                     'num_synthetic_metadata': 500 if 'synthmed' in exp_name else 0,
                     'num_synthetic_images': 500 if 'synthmed' in exp_name else 0,
                     'augmentation_strength': 'heavy' if exp_name == 'A8_geometric_only' else 'default'},
            'schema': {'schema_path': 'config/schema/clinical_metadata.json',
                       'repair_enabled': True if 'synthmed' in exp_name else False,
                       'repair_max_iterations': 3},
            'retrieval': {'embedder_model': 'sentence-transformers/all-MiniLM-L6-v2',
                          'top_k': 5, 'fusion_weights': [0.4, 0.3, 0.3],
                          'index_path': 'outputs/models/faiss_index.bin',
                          'rag_enabled': True if 'synthmed' in exp_name else False},
            'generation': {'metadata_model': 'distilgpt2', 'metadata_max_length': 256,
                           'temperature': 0.7, 'diffusion_timesteps': 100,
                           'diffusion_image_size': 32,
                           'diffusion_checkpoint': 'outputs/models/diffusion_unet.pt',
                           'diffusion_epochs': 20, 'conditioning_enabled': False},
            'classifier': {'model_name': 'mobilenet_v2', 'num_classes': 5,
                           'batch_size': 8, 'epochs': 50,
                           'learning_rate': 0.0001, 'weight_decay': 0.001,
                           'use_metadata': False},
            'evaluation': {'metrics': ['accuracy', 'f1', 'roc_auc'],
                           'save_results': True, 'results_path': 'outputs/results'}
        }
        config_path = f"config/exp_{exp_name}_seed{seed}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        
        start = time.time()
        try:
            run_pipeline(config_path)
            row = pd.read_csv("outputs/results/experiment_comparison.csv").iloc[-1]
            all_results.append({'experiment': exp_name, 'seed': seed,
                                'accuracy': row['accuracy'], 'f1_score': row['f1_score'],
                                'roc_auc': row['roc_auc']})
            logger.info(f"{exp_name} seed {seed}: acc={row['accuracy']}")
        except Exception as e:
            logger.error(f"{exp_name} seed {seed} failed: {e}")
    
# compute stats
df = pd.DataFrame(all_results)
summary = df.groupby('experiment').agg({'accuracy': ['mean', 'std'],
                                        'f1_score': ['mean', 'std'],
                                        'roc_auc': ['mean', 'std']})
print(summary)

a1 = df[df.experiment == 'A1_baseline_100real'].sort_values('seed')['accuracy'].values
a2 = df[df.experiment == 'A2_synthmed_100real_500syn'].sort_values('seed')['accuracy'].values
tstat, pval = ttest_rel(a2, a1)
print(f"Paired t-test (A2 vs A1): t={tstat:.4f}, p={pval:.4f}")

if 'A8_geometric_only' in df.experiment.values:
    a8 = df[df.experiment == 'A8_geometric_only'].sort_values('seed')['accuracy'].values
    tstat8, pval8 = ttest_rel(a8, a1)
    print(f"Paired t-test (A8 vs A1): t={tstat8:.4f}, p={pval8:.4f}")
summary.to_csv("outputs/results/multiseed_summary.csv")
df.to_csv("outputs/results/multiseed_raw.csv", index=False)