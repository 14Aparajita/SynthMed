# save as: diagnose_and_fix.py
# Run from synthmed directory

import sys
from pathlib import Path
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import torch
from collections import Counter

print("="*70)
print("SYNTHMED DIAGNOSTIC REPORT")
print("="*70)

# 1. Check data splits
print("\n📁 1. DATA SPLIT ANALYSIS")
print("-"*40)

clinical_path = Path("data/raw/clinical.csv")
if clinical_path.exists():
    df = pd.read_csv(clinical_path)
    print(f"Total records: {len(df)}")
    
    if 'dr_grade' in df.columns:
        grade_counts = df['dr_grade'].value_counts().sort_index()
        print(f"\nDR Grade distribution:")
        for grade, count in grade_counts.items():
            print(f"  Grade {grade}: {count} images ({count/len(df)*100:.1f}%)")
        
        if 'split' in df.columns:
            print(f"\nSplit distribution:")
            for split in ['train', 'val', 'test']:
                split_df = df[df['split'] == split]
                if len(split_df) > 0:
                    print(f"  {split}: {len(split_df)} images")
                    grades = split_df['dr_grade'].value_counts().sort_index()
                    print(f"    Grades: {dict(grades)}")
        else:
            print("\n⚠️ No 'split' column found! Creating proper splits...")
            create_proper_splits(df)
    else:
        print("❌ No 'dr_grade' column found!")
else:
    print("❌ clinical.csv not found!")

# 2. Check processed files
print("\n📁 2. PROCESSED FILES CHECK")
print("-"*40)

processed_dir = Path("data/processed")
npy_files = list(processed_dir.glob("*.npy"))
print(f"Processed files: {len(npy_files)}")

if npy_files:
    # Check file sizes
    sizes = [f.stat().st_size for f in npy_files[:100]]
    print(f"Average file size: {np.mean(sizes)/1024:.1f} KB")
    
    # Check for all-zero files
    zero_files = 0
    for f in npy_files[:50]:
        try:
            img = np.load(f)
            if img.max() == 0:
                zero_files += 1
        except:
            pass
    
    if zero_files > 0:
        print(f"⚠️ {zero_files} files are all zeros (corrupted)")
    
    # Check image stats
    try:
        sample = np.load(str(npy_files[0]))
        print(f"Sample shape: {sample.shape}")
        print(f"Value range: [{sample.min():.4f}, {sample.max():.4f}]")
        print(f"Mean: {sample.mean():.4f}, Std: {sample.std():.4f}")
    except Exception as e:
        print(f"Error loading sample: {e}")

# 3. Check for data leakage
print("\n📁 3. DATA LEAKAGE CHECK")
print("-"*40)

# Check if train and test have same images
processed_dir = Path("data/processed")
all_files = {f.stem for f in processed_dir.glob("*.npy")}

if 'df' in locals() and 'split' in df.columns:
    train_files = set()
    test_files = set()
    
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    if 'image_id' in df.columns:
        train_files = set(train_df['image_id'].values)
        test_files = set(test_df['image_id'].values)
    elif 'image_path' in df.columns:
        train_files = {Path(p).stem for p in train_df['image_path'] if pd.notna(p)}
        test_files = {Path(p).stem for p in test_df['image_path'] if pd.notna(p)}
    
    overlap = train_files & test_files
    if len(overlap) > 0:
        print(f"❌ DATA LEAKAGE: {len(overlap)} images in both train and test!")
        print(f"   Examples: {list(overlap)[:5]}")
    else:
        print(f"✅ No overlap: {len(train_files)} train, {len(test_files)} test")
    
    # Check class balance in test
    if len(test_df) > 0 and 'dr_grade' in test_df.columns:
        test_grades = test_df['dr_grade'].unique()
        print(f"Test set classes: {sorted(test_grades)}")
        if len(test_grades) == 1:
            print(f"❌ TEST SET HAS ONLY ONE CLASS: {test_grades[0]}")
            print("   This explains 100% accuracy and 0 ROC-AUC!")

# 4. Fix recommendations
print("\n" + "="*70)
print("RECOMMENDED FIXES")
print("="*70)

print("""
1. REPROCESS DATA WITH PROPER SPLITS:
   python fix_data_splits.py

2. DELETE CORRUPTED PROCESSED FILES:
   python clean_processed.py

3. RE-RUN WITH REALISTIC CONFIG:
   python experiments/run_pipeline.py --config config/default.yaml

4. EXPECTED HEALTHY RESULTS:
   - Accuracy: 0.60-0.85 (for 5-class DR)
   - F1 Score: 0.55-0.80
   - ROC-AUC: 0.75-0.95
   - Loss starting at 1.5-2.0, decreasing to 0.3-0.8
   - Schema validity: 0.85-1.00
""")

print("="*70)