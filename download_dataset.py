#!/usr/bin/env python3
"""
Download and organize the Chest CT Segmentation dataset from Kaggle.
"""
import os
import kagglehub
from pathlib import Path
import shutil

# Set Kaggle API token
os.environ["KAGGLE_API_TOKEN"] = "KGAT_5df2b06df133abf2ab9a49d25225d655"

# Project root
PROJECT_ROOT = Path("/Users/edwin/Desktop/Business/Technological/fl-sam2-lora")
DATASET_DIR = PROJECT_ROOT / "dataset" / "chest-ct-segmentation"

print("=" * 60)
print("Downloading Chest CT Segmentation Dataset from Kaggle")
print("=" * 60)

# Download dataset
print("\n1. Downloading dataset...")
path = kagglehub.dataset_download("polomarco/chest-ct-segmentation")
print(f"   Downloaded to: {path}")

# Check structure
print("\n2. Checking dataset structure...")
downloaded_path = Path(path)
print(f"   Contents: {list(downloaded_path.iterdir())}")

# Create target directory
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Check if we need to reorganize
images_dir = downloaded_path / "images"
masks_dir = downloaded_path / "masks"

if images_dir.exists() and masks_dir.exists():
    print("\n3. Organizing dataset structure...")
    
    # Check if nested structure exists (images/images, masks/masks)
    nested_images = images_dir / "images"
    nested_masks = masks_dir / "masks"
    
    if nested_images.exists() and nested_masks.exists():
        print("   Found nested structure (images/images, masks/masks)")
        # Copy to target location
        target_images = DATASET_DIR / "images" / "images"
        target_masks = DATASET_DIR / "masks" / "masks"
        target_images.parent.mkdir(parents=True, exist_ok=True)
        target_masks.parent.mkdir(parents=True, exist_ok=True)
        
        if not target_images.exists():
            print("   Copying images...")
            shutil.copytree(nested_images, target_images)
        else:
            print("   Images already exist, skipping...")
            
        if not target_masks.exists():
            print("   Copying masks...")
            shutil.copytree(nested_masks, target_masks)
        else:
            print("   Masks already exist, skipping...")
    else:
        # Flat structure - create nested structure
        print("   Found flat structure, creating nested structure...")
        target_images = DATASET_DIR / "images" / "images"
        target_masks = DATASET_DIR / "masks" / "masks"
        target_images.parent.mkdir(parents=True, exist_ok=True)
        target_masks.parent.mkdir(parents=True, exist_ok=True)
        
        if not target_images.exists():
            print("   Copying images...")
            shutil.copytree(images_dir, target_images)
        else:
            print("   Images already exist, skipping...")
            
        if not target_masks.exists():
            print("   Copying masks...")
            shutil.copytree(masks_dir, target_masks)
        else:
            print("   Masks already exist, skipping...")
    
    # Copy train.csv if it exists
    train_csv = downloaded_path / "train.csv"
    if train_csv.exists():
        target_csv = DATASET_DIR / "train.csv"
        if not target_csv.exists():
            print("   Copying train.csv...")
            shutil.copy2(train_csv, target_csv)
        else:
            print("   train.csv already exists, skipping...")
    
    print(f"\n✓ Dataset organized at: {DATASET_DIR}")
    print(f"   Images: {len(list((DATASET_DIR / 'images' / 'images').glob('*.jpg')))} files")
    print(f"   Masks: {len(list((DATASET_DIR / 'masks' / 'masks').glob('*.jpg')))} files")
    
    if (DATASET_DIR / "train.csv").exists():
        print(f"   train.csv: ✓")
else:
    print(f"\n⚠ Warning: Expected 'images' and 'masks' directories not found in {downloaded_path}")
    print(f"   Please check the dataset structure manually.")

print("\n" + "=" * 60)
print("Dataset download complete!")
print("=" * 60)

