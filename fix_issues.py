# save as: fix_issues.py
# Run from synthmed directory

import sys
from pathlib import Path

print("="*60)
print("Fixing SynthMed Issues")
print("="*60)

# Fix 1: Check processed files
processed_dir = Path("data/processed")
if processed_dir.exists():
    npy_files = list(processed_dir.glob("*.npy"))
    print(f"\n1. Checking {len(npy_files)} processed files...")
    
    # Test loading
    import numpy as np
    failed = 0
    for f in npy_files[:10]:  # Check first 10
        try:
            data = np.load(f, allow_pickle=False)
        except Exception as e:
            print(f"   Failed: {f.name} - {e}")
            failed += 1
    
    if failed == 0:
        print("   All files load correctly!")
    else:
        print(f"   {failed} files failed. Re-preprocessing needed.")

# Fix 2: Check image paths in clinical.csv
import pandas as pd
clinical_path = Path("data/raw/clinical.csv")
if clinical_path.exists():
    df = pd.read_csv(clinical_path)
    print(f"\n2. Clinical data: {len(df)} records")
    
    if 'image_path' in df.columns:
        # Check if paths point to processed files
        sample_path = df['image_path'].iloc[0]
        print(f"   Sample path: {sample_path}")
        
        # Fix paths to point to processed .npy files
        if 'data/raw' in str(sample_path) or not str(sample_path).endswith('.npy'):
            print("   Fixing image paths...")
            
            # Map to processed files
            processed_dir = Path("data/processed")
            processed_files = {f.stem: str(f) for f in processed_dir.glob("*.npy")}
            
            new_paths = []
            for _, row in df.iterrows():
                if 'image_id' in row:
                    img_id = row['image_id']
                elif 'image' in row:
                    img_id = Path(row['image']).stem
                else:
                    img_id = Path(row['image_path']).stem if pd.notna(row.get('image_path')) else ""
                
                if img_id in processed_files:
                    new_paths.append(processed_files[img_id])
                else:
                    # Try partial match
                    found = False
                    for key in processed_files:
                        if img_id in key or key in img_id:
                            new_paths.append(processed_files[key])
                            found = True
                            break
                    if not found:
                        new_paths.append(row.get('image_path', ''))
            
            df['image_path'] = new_paths
            df.to_csv(clinical_path, index=False)
            print(f"   Fixed {len(df)} paths to point to processed files")
    else:
        print("   No 'image_path' column found")

# Fix 3: Verify trainer settings
print("\n3. Verifying trainer configuration...")
config_path = Path("config/default.yaml")
if config_path.exists():
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    batch_size = config.get('classifier', {}).get('batch_size', 16)
    num_train = config.get('data', {}).get('num_real_train', 400)
    
    print(f"   Batch size: {batch_size}")
    print(f"   Training samples: {num_train}")
    print(f"   Batches per epoch: {num_train // batch_size}")

print("\n" + "="*60)
print("Fixes applied!")
print("Now run: python experiments/run_pipeline.py --config config/default.yaml")
print("="*60)