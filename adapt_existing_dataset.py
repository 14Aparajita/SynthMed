# save as: adapt_existing_dataset.py
# Run this from the synthmed directory

import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import random

# Your existing dataset path
EXISTING_DATASET_PATH = r"C:\MyFolders\Projects\major_project\classification\dataset\aptos_resized_kaggle\split_data"

# Class mapping
CLASS_MAPPING = {
    'No_DR': 0,
    'Mild': 1,
    'Moderate': 2,
    'Severe': 3,
    'Proliferate_DR': 4
}

def adapt_existing_dataset():
    """Adapt your existing dataset to SynthMed format."""
    
    existing_path = Path(EXISTING_DATASET_PATH)
    synthmed_raw = Path("data/raw")
    synthmed_raw.mkdir(parents=True, exist_ok=True)
    
    print(f"Looking for dataset at: {existing_path}")
    print(f"Folders found: {[d.name for d in existing_path.iterdir() if d.is_dir()]}")
    
    all_records = []
    image_counter = 0
    
    # Process train/val/test splits
    for split in ['train', 'val', 'test']:
        split_path = existing_path / split
        
        if not split_path.exists():
            print(f"⚠️  Split folder not found: {split_path}")
            continue
        
        print(f"\n📁 Processing {split} split...")
        
        for class_name, class_label in CLASS_MAPPING.items():
            class_path = split_path / class_name
            
            if not class_path.exists():
                print(f"  ⚠️  Class folder not found: {class_path}")
                continue
            
            # Get all images in this class
            image_files = list(class_path.glob('*.png')) + \
                         list(class_path.glob('*.jpg')) + \
                         list(class_path.glob('*.jpeg'))
            
            print(f"  📸 {class_name}: {len(image_files)} images")
            
            for img_path in image_files:
                # Copy image to SynthMed raw directory with new name
                new_name = f"img_{image_counter:05d}{img_path.suffix}"
                dest_path = synthmed_raw / new_name
                
                # Copy the image
                shutil.copy2(img_path, dest_path)
                
                # Record metadata
                all_records.append({
                    'image_id': f"img_{image_counter:05d}",
                    'image_path': str(dest_path),
                    'dr_grade': class_label,
                    'class_name': class_name,
                    'split': split,
                    'original_path': str(img_path)
                })
                
                image_counter += 1
    
    # Create clinical metadata CSV
    print(f"\n✅ Copied {image_counter} images total")
    
    # Add synthetic clinical metadata
    np.random.seed(42)
    clinical_data = []
    
    for record in all_records:
        dr_grade = record['dr_grade']
        
        # Generate realistic clinical metadata correlated with DR grade
        age = np.random.randint(25, 85)
        
        # Higher grades tend to have older patients
        if dr_grade >= 3:
            age = np.random.randint(45, 85)
        elif dr_grade >= 1:
            age = np.random.randint(35, 75)
        
        clinical_record = {
            'image_id': record['image_id'],
            'image_path': record['image_path'],
            'dr_grade': dr_grade,
            'split': record['split'],
            'patient_id': f"P{np.random.randint(10000, 99999):05d}",
            'age': age,
            'sex': np.random.choice(['M', 'F']),
            'image_quality': round(np.random.uniform(0.5, 1.0), 2),
            'left_eye': np.random.choice([True, False]),
            'microaneurysms': _get_finding(dr_grade, 'microaneurysms'),
            'hemorrhages': _get_finding(dr_grade, 'hemorrhages'),
            'exudates': _get_finding(dr_grade, 'exudates'),
        }
        clinical_data.append(clinical_record)
    
    # Save clinical CSV
    clinical_df = pd.DataFrame(clinical_data)
    clinical_df.to_csv(synthmed_raw / 'clinical.csv', index=False)
    
    # Save train/test split info
    train_df = clinical_df[clinical_df['split'] == 'train']
    val_df = clinical_df[clinical_df['split'] == 'val']
    test_df = clinical_df[clinical_df['split'] == 'test']
    
    # If no val split, use part of train
    if len(val_df) == 0:
        train_df, val_df = train_test_split_custom(train_df, test_size=0.15)
    
    train_df.to_csv(synthmed_raw / 'train.csv', index=False)
    val_df.to_csv(synthmed_raw / 'val.csv', index=False)
    test_df.to_csv(synthmed_raw / 'test.csv', index=False)
    
    # Print statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total images: {len(clinical_df)}")
    print(f"\nSplit distribution:")
    for split in ['train', 'val', 'test']:
        count = len(clinical_df[clinical_df['split'] == split])
        print(f"  {split}: {count} images")
    
    print(f"\nClass distribution:")
    for class_name, class_label in CLASS_MAPPING.items():
        count = len(clinical_df[clinical_df['dr_grade'] == class_label])
        print(f"  {class_name} (Grade {class_label}): {count} images")
    
    print(f"\n✅ Dataset adapted successfully!")
    print(f"📂 Images saved to: {synthmed_raw}")
    print(f"📊 Metadata saved to: {synthmed_raw / 'clinical.csv'}")
    
    return clinical_df

def _get_finding(dr_grade, finding_type):
    """Get realistic anatomical finding based on DR grade."""
    if dr_grade == 0:
        return 'none'
    elif dr_grade == 1:
        probs = {'microaneurysms': [0.2, 0.6, 0.15, 0.05],
                 'hemorrhages': [0.7, 0.2, 0.08, 0.02],
                 'exudates': [0.8, 0.15, 0.04, 0.01]}
    elif dr_grade == 2:
        probs = {'microaneurysms': [0.05, 0.15, 0.5, 0.3],
                 'hemorrhages': [0.1, 0.4, 0.35, 0.15],
                 'exudates': [0.2, 0.4, 0.3, 0.1]}
    elif dr_grade == 3:
        probs = {'microaneurysms': [0.01, 0.05, 0.24, 0.7],
                 'hemorrhages': [0.02, 0.1, 0.38, 0.5],
                 'exudates': [0.05, 0.15, 0.4, 0.4]}
    else:  # Grade 4
        probs = {'microaneurysms': [0.0, 0.02, 0.18, 0.8],
                 'hemorrhages': [0.0, 0.05, 0.25, 0.7],
                 'exudates': [0.01, 0.1, 0.39, 0.5]}
    
    levels = ['none', 'few', 'moderate', 'many']
    return np.random.choice(levels, p=probs.get(finding_type, [0.25]*4))

def train_test_split_custom(df, test_size=0.15):
    """Custom stratified split."""
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, test_size=test_size, 
        stratify=df['dr_grade'],
        random_state=42
    )
    return train_df, val_df

if __name__ == "__main__":
    # Check if path exists
    if not Path(EXISTING_DATASET_PATH).exists():
        print(f"❌ Dataset path not found: {EXISTING_DATASET_PATH}")
        print("\nPlease update EXISTING_DATASET_PATH in this script")
        print("to point to your actual dataset location.")
        exit(1)
    
    adapt_existing_dataset()