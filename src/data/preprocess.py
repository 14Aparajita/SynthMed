# src/data/preprocess.py - FIXED VERSION

import cv2
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import logging

logger = logging.getLogger("synthmed.data")

def preprocess_images(
    input_dir: str,
    output_dir: str,
    image_size: int = 128,
    use_clahe: bool = True
) -> List[str]:
    """
    Preprocess retinal images: resize, CLAHE enhancement, normalize.
    Works with your existing APTOS dataset.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    processed_files = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.JPG', '.JPEG', '.PNG'}
    
    # Find all images - search recursively
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_path.glob(f'*{ext}')))
    
    if not image_files:
        # Try looking in subdirectories
        logger.info("No images in root, searching subdirectories...")
        for ext in image_extensions:
            image_files.extend(list(input_path.glob(f'**/*{ext}')))
    
    if not image_files:
        logger.warning(f"No images found in {input_dir}")
        return []
    
    logger.info(f"Found {len(image_files)} images to preprocess")
    
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    else:
        clahe = None
    
    for img_file in tqdm(image_files, desc="Preprocessing images"):
        try:
            # Read image
            img = cv2.imread(str(img_file))
            if img is None:
                logger.warning(f"Could not read: {img_file}")
                continue
            
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, (image_size, image_size))
            
            # Apply CLAHE for better contrast (important for DR detection)
            if clahe is not None:
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                l_channel = clahe.apply(l_channel)
                lab = cv2.merge([l_channel, a_channel, b_channel])
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Save as numpy with allow_pickle=True for compatibility
            out_file = output_path / f"{img_file.stem}.npy"
            np.save(out_file, img, allow_pickle=False)  # Standard save, no pickle needed
            
            processed_files.append(str(out_file))
            
        except Exception as e:
            logger.error(f"Error processing {img_file}: {e}")
            continue
    
    logger.info(f"Preprocessed {len(processed_files)} images")
    return processed_files

def load_clinical_data(
    clinical_path: str,
    image_mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Load clinical metadata, mapping to image files.
    """
    clinical_file = Path(clinical_path)
    
    if clinical_file.exists():
        logger.info(f"Loading clinical data from {clinical_path}")
        df = pd.read_csv(clinical_path)
        
        # Ensure required columns exist
        if 'dr_grade' not in df.columns:
            # Try common column names
            for col in ['diagnosis', 'level', 'label', 'class']:
                if col in df.columns:
                    df['dr_grade'] = df[col]
                    break
        
        if 'image_path' not in df.columns:
            if 'image_id' in df.columns:
                df['image_path'] = df['image_id'].apply(
                    lambda x: _find_image_path(x, image_mapping)
                )
            elif 'image' in df.columns:
                df['image_path'] = df['image'].apply(
                    lambda x: _find_image_path(x, image_mapping)
                )
        
        # Filter to only images that exist
        valid_mask = df['image_path'].apply(
            lambda x: Path(x).exists() if x else False
        )
        
        if not valid_mask.all():
            n_missing = (~valid_mask).sum()
            logger.warning(f"{n_missing} images not found, filtering...")
            df = df[valid_mask].reset_index(drop=True)
        
    else:
        logger.warning(f"No clinical data found at {clinical_path}")
        # Create from image files
        df = _create_clinical_from_images(image_mapping)
    
    return df

def _find_image_path(image_id: str, image_mapping: Dict[str, str]) -> str:
    """Find image path from mapping."""
    # Try exact match first
    if image_id in image_mapping:
        return image_mapping[image_id]
    
    # Try without extension
    image_id_no_ext = Path(image_id).stem
    for key, path in image_mapping.items():
        if image_id_no_ext in key or key in image_id_no_ext:
            return path
    
    # Try with .npy extension
    npy_key = f"{image_id_no_ext}.npy"
    if npy_key in image_mapping:
        return image_mapping[npy_key]
    
    return ""

def _create_clinical_from_images(image_mapping: Dict[str, str]) -> pd.DataFrame:
    """Create clinical dataframe from image files only."""
    records = []
    
    for img_path_str in image_mapping.values():
        img_path = Path(img_path_str)
        img_name = img_path.stem
        
        # Try to extract DR grade from path
        grade = 0
        path_lower = str(img_path).lower()
        
        for class_name, class_grade in {
            'no_dr': 0, 'none': 0, 'normal': 0, '0': 0,
            'mild': 1, '1': 1,
            'moderate': 2, 'mod': 2, '2': 2,
            'severe': 3, 'sev': 3, '3': 3,
            'proliferative': 4, 'prolif': 4, 'pdr': 4, '4': 4
        }.items():
            if class_name in path_lower or class_name in img_name.lower():
                grade = class_grade
                break
        
        records.append({
            'image_id': img_name,
            'image_path': img_path_str,
            'dr_grade': grade,
        })
    
    return pd.DataFrame(records)