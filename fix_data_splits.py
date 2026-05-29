# save as: fix_data_splits.py
# Creates proper stratified train/test splits

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

print("="*60)
print("FIXING DATA SPLITS FOR SYNTHMED")
print("="*60)

# Load clinical data
df = pd.read_csv("data/raw/clinical.csv")
print(f"\nOriginal data: {len(df)} records")

# Check if dr_grade exists
if 'dr_grade' not in df.columns:
    print("❌ No dr_grade column! Creating from existing columns...")
    
    # Try to infer from class folder names in original path
    if 'original_path' in df.columns:
        class_mapping = {
            'No_DR': 0, 'no_dr': 0, 'normal': 0,
            'Mild': 1, 'mild': 1,
            'Moderate': 2, 'moderate': 2,
            'Severe': 3, 'severe': 3,
            'Proliferate_DR': 4, 'proliferative': 4
        }
        
        def extract_grade(path):
            path_lower = str(path).lower()
            for class_name, grade in class_mapping.items():
                if class_name.lower() in path_lower:
                    return grade
            return 0
        
        df['dr_grade'] = df['original_path'].apply(extract_grade)
    else:
        # Assign random grades for demo
        print("⚠️ Assigning synthetic grades for demonstration")
        df['dr_grade'] = np.random.choice([0,1,2,3,4], len(df))

print(f"\nClass distribution:")
for grade in sorted(df['dr_grade'].unique()):
    count = (df['dr_grade'] == grade).sum()
    print(f"  Grade {grade}: {count} ({count/len(df)*100:.1f}%)")

# Identify image ID column
id_col = None
for col in ['image_id', 'image', 'id_code']:
    if col in df.columns:
        id_col = col
        break

if id_col is None:
    print("❌ No image ID column found!")
    exit(1)

# Create stratified splits (70/15/15)
print(f"\nCreating stratified splits...")

# First, split off test set (15%)
train_val, test = train_test_split(
    df, 
    test_size=0.15, 
    stratify=df['dr_grade'],
    random_state=42
)

# Then split remaining into train (70% of total) and val (15% of total)
train, val = train_test_split(
    train_val,
    test_size=0.15/0.85,  # 15% of total from 85% remaining
    stratify=train_val['dr_grade'],
    random_state=42
)

# Assign split labels
df['split'] = 'train'
df.loc[df[id_col].isin(val[id_col]), 'split'] = 'val'
df.loc[df[id_col].isin(test[id_col]), 'split'] = 'test'

print(f"\nSplit distribution:")
for split in ['train', 'val', 'test']:
    split_df = df[df['split'] == split]
    print(f"\n{split.upper()} ({len(split_df)} images):")
    for grade in sorted(split_df['dr_grade'].unique()):
        count = (split_df['dr_grade'] == grade).sum()
        print(f"  Grade {grade}: {count}")

# Verify no overlap
train_ids = set(df[df['split'] == 'train'][id_col])
test_ids = set(df[df['split'] == 'test'][id_col])
val_ids = set(df[df['split'] == 'val'][id_col])

assert len(train_ids & test_ids) == 0, "Train-Test overlap detected!"
assert len(train_ids & val_ids) == 0, "Train-Val overlap detected!"
assert len(test_ids & val_ids) == 0, "Test-Val overlap detected!"
print("\n✅ No data leakage detected!")

# Save updated clinical data
df.to_csv("data/raw/clinical.csv", index=False)
df[df['split'] == 'train'].to_csv("data/raw/train.csv", index=False)
df[df['split'] == 'val'].to_csv("data/raw/val.csv", index=False)
df[df['split'] == 'test'].to_csv("data/raw/test.csv", index=False)

print(f"\n✅ Updated clinical data saved!")
print(f"   Train: {len(train)} images")
print(f"   Val: {len(val)} images")
print(f"   Test: {len(test)} images")

# Update config to use proper numbers
import yaml

config_path = Path("config/default.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

config['data']['num_real_train'] = len(train)
config['data']['num_real_test'] = len(test)

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"\n✅ Updated config with correct split sizes")
print(f"\nNow run: python experiments/run_pipeline.py --config config/default.yaml")