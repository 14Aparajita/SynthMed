# save as: setup_with_existing_data.py
# Run this script to set up SynthMed with your existing APTOS dataset

import os
import sys
from pathlib import Path
import shutil

# Your existing dataset path
EXISTING_DATASET = r"C:\MyFolders\Projects\major_project\classification\dataset\aptos_resized_kaggle\split_data"

def setup_synthmed_with_existing_data():
    """Set up SynthMed using your existing APTOS dataset."""
    
    print("="*60)
    print("SynthMed Setup with Existing APTOS Dataset")
    print("="*60)
    
    # Check if existing dataset exists
    existing_path = Path(EXISTING_DATASET)
    if not existing_path.exists():
        print(f"❌ Dataset not found at: {EXISTING_DATASET}")
        print("\nPlease update the EXISTING_DATASET path in this script.")
        return
    
    print(f"✅ Found existing dataset at: {EXISTING_DATASET}")
    
    # Create project structure
    print("\n📁 Creating project structure...")
    directories = [
        "config/schema",
        "data/raw", 
        "data/processed",
        "data/knowledge_base",
        "src/data",
        "src/schema", 
        "src/retrieval", 
        "src/generation",
        "src/classifier", 
        "src/evaluation", 
        "src/utils",
        "experiments", 
        "scripts", 
        "tests",
        "outputs/models",
        "outputs/synthetic",
        "outputs/results",
        "outputs/logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Create __init__.py in Python packages
        if directory.startswith("src") or directory == "experiments" or directory == "tests":
            init_file = Path(directory) / "__init__.py"
            if not init_file.exists():
                init_file.touch()
    
    print("✅ Project structure created")
    
    # Adapt existing dataset
    print("\n🔄 Adapting your existing dataset...")
    print("This will copy images to data/raw/ and create metadata...")
    
    # Count images in existing dataset
    total_images = 0
    for split in ['train', 'val', 'test']:
        split_path = existing_path / split
        if split_path.exists():
            for class_folder in split_path.iterdir():
                if class_folder.is_dir():
                    images = list(class_folder.glob('*'))
                    total_images += len(images)
    
    print(f"Found {total_images} images in your dataset")
    
    # Copy first few images as sample (to save space)
    # If you want all images, remove the sample limit
    use_all_images = input("\nUse ALL images? (y/n, default: n for quick setup): ").lower() == 'y'
    
    max_per_class = None if use_all_images else 50  # Limit for quick testing
    
    import_images_from_existing(existing_path, max_per_class)
    
    print("\n" + "="*60)
    print("✅ Setup complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. cd synthmed")
    print("2. pip install -r requirements.txt")
    print("3. python scripts/build_kb.py")
    print("4. python experiments/run_pipeline.py")
    
    return True

def import_images_from_existing(existing_path, max_per_class=None):
    """Import images from existing dataset."""
    from PIL import Image
    import pandas as pd
    import numpy as np
    
    class_mapping = {
        'No_DR': 0,
        'Mild': 1, 
        'Moderate': 2,
        'Severe': 3,
        'Proliferate_DR': 4
    }
    
    raw_dir = Path("data/raw")
    all_records = []
    counter = 0
    
    for split in ['train', 'val', 'test']:
        split_path = existing_path / split
        
        if not split_path.exists():
            continue
        
        print(f"\nProcessing {split} split...")
        
        for class_name, class_label in class_mapping.items():
            class_path = split_path / class_name
            
            if not class_path.exists():
                continue
            
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_files.extend(list(class_path.glob(ext)))
            
            # Limit per class if specified
            if max_per_class:
                image_files = image_files[:max_per_class]
            
            print(f"  {class_name}: {len(image_files)} images")
            
            for img_file in image_files:
                try:
                    # Read and resize image
                    img = Image.open(img_file).convert('RGB')
                    img = img.resize((128, 128), Image.Resampling.LANCZOS)
                    
                    # Save to raw directory
                    new_name = f"img_{counter:05d}.png"
                    img.save(raw_dir / new_name)
                    
                    # Record metadata
                    all_records.append({
                        'image_id': f"img_{counter:05d}",
                        'image_path': str(raw_dir / new_name),
                        'dr_grade': class_label,
                        'split': split,
                        'original_class': class_name
                    })
                    
                    counter += 1
                    
                except Exception as e:
                    print(f"    ⚠️ Error with {img_file}: {e}")
                    continue
    
    # Create clinical metadata
    np.random.seed(42)
    clinical_records = []
    
    for record in all_records:
        dr_grade = record['dr_grade']
        
        # Generate age based on DR severity
        if dr_grade >= 3:
            age = np.random.randint(50, 85)
        elif dr_grade >= 1:
            age = np.random.randint(35, 75)
        else:
            age = np.random.randint(25, 70)
        
        clinical_records.append({
            **record,
            'patient_id': f"P{np.random.randint(10000, 99999)}",
            'age': age,
            'sex': np.random.choice(['M', 'F']),
            'image_quality': round(np.random.uniform(0.6, 1.0), 2),
            'left_eye': np.random.choice([True, False]),
        })
    
    # Save clinical CSV
    df = pd.DataFrame(clinical_records)
    df.to_csv(raw_dir / 'clinical.csv', index=False)
    
    # Create separate train/val/test CSVs
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        if len(split_df) > 0:
            split_df.to_csv(raw_dir / f'{split}.csv', index=False)
    
    print(f"\n✅ Imported {counter} images total")
    print(f"📊 Metadata saved to data/raw/clinical.csv")
    
    return df

if __name__ == "__main__":
    setup_synthmed_with_existing_data()