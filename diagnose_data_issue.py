# save as: diagnose_data_issue.py
# Run this to see EXACTLY what's in your data

import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("DATA DIAGNOSTIC - Finding the single-class issue")
print("="*70)

# 1. Check clinical.csv
print("\n1. CLINICAL.CSV ANALYSIS")
df = pd.read_csv("data/raw/clinical.csv")
print(f"   Total records: {len(df)}")
print(f"   Columns: {df.columns.tolist()}")

if 'dr_grade' in df.columns:
    print(f"\n   DR Grade distribution:")
    for grade in sorted(df['dr_grade'].unique()):
        count = (df['dr_grade'] == grade).sum()
        print(f"   Grade {grade}: {count} ({count/len(df)*100:.1f}%)")

if 'split' in df.columns:
    print(f"\n   Split distribution:")
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        if len(split_df) > 0:
            print(f"\n   {split.upper()}: {len(split_df)} records")
            grades = split_df['dr_grade'].value_counts().sort_index()
            print(f"   Grades: {dict(grades)}")

# 2. Check what run_pipeline actually uses
print("\n2. PIPELINE DATA USAGE")

# Read the config
import yaml
with open("config/improved.yaml" if Path("config/improved.yaml").exists() else "config/default.yaml") as f:
    config = yaml.safe_load(f)

n_train = config['data']['num_real_train']
n_test = config['data']['num_real_test']
print(f"   Config says: train={n_train}, test={n_test}")

# Check how pipeline splits data
print(f"\n   Pipeline takes first {n_train} for train")
print(f"   Pipeline takes next {n_test} for test")
print(f"   If first {n_train} rows are all same class -> PROBLEM!")

# Show what classes are in first N rows
if 'dr_grade' in df.columns:
    first_train = df.iloc[:n_train]
    first_test = df.iloc[n_train:n_train+n_test]
    
    print(f"\n   First {n_train} rows (train):")
    print(f"   Classes: {first_train['dr_grade'].value_counts().to_dict()}")
    
    print(f"\n   Next {n_test} rows (test):")
    print(f"   Classes: {first_test['dr_grade'].value_counts().to_dict()}")

# 3. Check if data is sorted by class
print("\n3. DATA ORDER CHECK")
if 'dr_grade' in df.columns:
    print(f"   First 10 grades: {df['dr_grade'].head(10).tolist()}")
    print(f"   Middle 10 grades: {df['dr_grade'].iloc[100:110].tolist()}")
    print(f"   Last 10 grades: {df['dr_grade'].tail(10).tolist()}")
    
    # Check if sorted
    is_sorted = df['dr_grade'].is_monotonic_increasing or df['dr_grade'].is_monotonic_decreasing
    if is_sorted:
        print("   ⚠️ DATA IS SORTED BY CLASS! This causes single-class splits!")
        print("   Solution: Shuffle the data before splitting")
    else:
        print("   Data appears randomly ordered (good)")