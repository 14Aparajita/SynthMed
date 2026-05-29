# save as: fix_data_properly.py
# This completely fixes the data split issue

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

print("="*70)
print("PROPER DATA SPLIT FIX")
print("="*70)

# Load and SHUFFLE the data
df = pd.read_csv("data/raw/clinical.csv")
print(f"\nOriginal data: {len(df)} records")

# SHUFFLE FIRST (this is critical!)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("✅ Data shuffled")

# Verify DR grade column exists
if 'dr_grade' not in df.columns:
    print("❌ No 'dr_grade' column! Creating from image paths...")
    
    # Try to extract from existing columns
    class_mapping = {
        'no_dr': 0, 'none': 0, 'normal': 0,
        'mild': 1, 
        'moderate': 2, 'mod': 2,
        'severe': 3, 'sev': 3,
        'proliferate_dr': 4, 'prolif': 4, 'pdr': 4
    }
    
    def extract_grade(row):
        # Check all string columns for class names
        for col in row.index:
            if isinstance(row[col], str):
                row_lower = row[col].lower()
                for class_name, grade in class_mapping.items():
                    if class_name in row_lower:
                        return grade
        return 0
    
    df['dr_grade'] = df.apply(extract_grade, axis=1)

# Ensure we have enough samples per class
grade_counts = df['dr_grade'].value_counts()
print(f"\nClass distribution after shuffle:")
for grade in sorted(grade_counts.index):
    print(f"  Grade {grade}: {grade_counts[grade]}")

# Create stratified splits
min_class_count = grade_counts.min()
if min_class_count < 10:
    print(f"\n⚠️ Some classes have very few samples. Minimum: {min_class_count}")
    print("   This may cause issues. Using minimum for stratification.")

# Split with stratification
try:
    # First split: separate test (15%)
    train_val, test = train_test_split(
        df, 
        test_size=0.15, 
        stratify=df['dr_grade'],
        random_state=42
    )
    
    # Second split: separate train and val
    train, val = train_test_split(
        train_val,
        test_size=0.176,  # 15% of total from remaining 85%
        stratify=train_val['dr_grade'],
        random_state=42
    )
    
except ValueError as e:
    print(f"\n⚠️ Stratification failed: {e}")
    print("   Using random split instead...")
    train_val, test = train_test_split(df, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.176, random_state=42)

# Assign split labels
df['split'] = 'train'
df.loc[df.index.isin(val.index), 'split'] = 'val'
df.loc[df.index.isin(test.index), 'split'] = 'test'

# Verify distributions
print(f"\n✅ FINAL SPLIT DISTRIBUTIONS:")
for split_name in ['train', 'val', 'test']:
    split_data = df[df['split'] == split_name]
    print(f"\n{split_name.upper()} ({len(split_data)} samples):")
    
    if 'dr_grade' in split_data.columns:
        for grade in sorted(split_data['dr_grade'].unique()):
            count = (split_data['dr_grade'] == grade).sum()
            print(f"  Grade {grade}: {count}")

# Verify no overlap
train_ids = set(df[df['split'] == 'train'].index)
val_ids = set(df[df['split'] == 'val'].index)
test_ids = set(df[df['split'] == 'test'].index)

assert len(train_ids & test_ids) == 0, "ERROR: Train-Test overlap!"
assert len(train_ids & val_ids) == 0, "ERROR: Train-Val overlap!"
print("\n✅ No data leakage detected!")

# Save the fixed data
df.to_csv("data/raw/clinical.csv", index=False)
print("✅ Saved fixed clinical.csv")

# Update config with correct sizes
import yaml

config_path = "config/improved.yaml" if Path("config/improved.yaml").exists() else "config/default.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

config['data']['num_real_train'] = len(train)
config['data']['num_real_test'] = len(test)

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"✅ Updated config: train={len(train)}, val={len(val)}, test={len(test)}")
print(f"\nNow run: python experiments/run_pipeline.py --config {config_path}")