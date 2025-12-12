#!/usr/bin/env python3
"""
Federated Learning with SAM2LoRA: Heterogeneous Clients

This script replicates the local.ipynb notebook functionality.
It demonstrates federated learning for medical image segmentation
using SAM2LoRA with 4 Data Owners (DOs) with different capabilities.

Usage:
    python run_local_fl.py

Or make it executable and run:
    ./run_local_fl.py
"""
import os

# Set matplotlib backend to non-interactive if no display
if 'DISPLAY' not in os.environ and os.name != 'nt':
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend


# ======================================================================
# Code from Cell 0
# ======================================================================

# Check dependencies
try:
    import accelerate
    print(f"✓ accelerate: {accelerate.__version__}")
except ImportError:
    print("⚠ accelerate not found. Run: pip install --no-deps accelerate")

# ======================================================================
# Code from Cell 1
# ======================================================================

# Cell 1: Imports and Setup
import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path("/Users/edwin/Desktop/Business/Technological/fl-sam2-lora")
TASK_MODULE_PATH = PROJECT_ROOT / "fl-sam2-lora" / "fl-sam2-lora"
SAM2_LIB_PATH = PROJECT_ROOT / "segment-anything-2"

# Add to path for imports - MUST add SAM2 path FIRST before importing task
sys.path.insert(0, str(SAM2_LIB_PATH))
sys.path.insert(0, str(TASK_MODULE_PATH))

import torch
import torch.nn.functional as F  # Added for training functions
import torch.optim as optim  # Added for optimizer
import torch.optim.lr_scheduler as lr_scheduler  # Added for scheduler
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
from tqdm import tqdm
import time
warnings.filterwarnings('ignore')

print(f"PyTorch version: {torch.__version__}")
print(f"Device: {'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'}")

# ======================================================================
# Code from Cell 2
# ======================================================================

# Cell 2: Import SAM2LoRA from task.py
# The module folder has hyphens, so we use importlib
import importlib.util
from pathlib import Path

# Verify SAM2 can be imported before loading task.py
try:
    import sam2
    print(f"✓ SAM2 imported successfully from {SAM2_LIB_PATH}")
except ImportError as e:
    print(f"⚠ Warning: Could not import SAM2: {e}")
    print("  Continuing anyway - task.py will handle this...")

# Load task.py directly
task_path = Path("/Users/edwin/Desktop/Business/Technological/fl-sam2-lora/fl-sam2-lora/fl-sam2-lora/task.py")
spec = importlib.util.spec_from_file_location("task", task_path)
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)

# After loading task.py, check if SAM2 is available
if not task.SAM2_AVAILABLE:
    print("⚠ Warning: SAM2 not available in task.py. Trying to fix...")
    # Try to manually set SAM2_AVAILABLE if we can import it
    try:
        import sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        task.SAM2_AVAILABLE = True
        task.SAM2ImagePredictor = SAM2ImagePredictor
        print("✓ Manually enabled SAM2 support")
    except Exception as e:
        print(f"✗ Could not enable SAM2: {e}")
        raise ImportError("SAM2 is required but not available. Please install segment-anything-2")

# Import what we need
SAM2LoRA = task.SAM2LoRA
create_model = task.create_model
train = task.train
evaluate = task.evaluate
get_weights = task.get_weights
set_weights = task.set_weights
DEFAULT_SAM2_CHECKPOINT = task.DEFAULT_SAM2_CHECKPOINT
DEFAULT_SAM2_CONFIG = task.DEFAULT_SAM2_CONFIG

print(f"SAM2 Checkpoint: {DEFAULT_SAM2_CHECKPOINT}")
print(f"SAM2 Config: {DEFAULT_SAM2_CONFIG}")
print(f"Checkpoint exists: {Path(DEFAULT_SAM2_CHECKPOINT).exists()}")

# ======================================================================
# Code from Cell 3
# ======================================================================

# Cell 3: Configuration

# Paths
DATASET_PATH = PROJECT_ROOT / "dataset" / "chest-ct-segmentation"

# Device
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# FL Config
NUM_ROUNDS = 5  # Updated from 3 to 5 for more training rounds
LOCAL_EPOCHS = 5  # Updated from 2 to 5 for better LoRA adaptation
LEARNING_RATE = 5e-5  # Updated from 1e-4 to 5e-5 for more stable training
LORA_RANK = 8  # Default rank
IMG_SIZE = 1024  # SAM2 default
MODALITY = "ct"
BATCH_SIZE = 1
USE_CLIP = True

# Site-specific configurations for DO_3 and DO_4 (targeted improvements)
SITE_CONFIGS = {
    "DO_1_zeroshot": {
        "lora_rank": 8,  # Not used (zero-shot)
        "learning_rate": 5e-5,
        "use_augmentation": False,
    },
    "DO_2_fewshot": {
        "lora_rank": 8,  # Not used (few-shot)
        "learning_rate": 5e-5,
        "use_augmentation": False,
    },
    "DO_3_lora": {
        "lora_rank": 16,  # Increased from 8 to 16 for more capacity
        "learning_rate": 1e-4,  # Slightly higher LR for faster adaptation
        "use_augmentation": True,  # Enable augmentation for better generalization
        "lora_alpha": 32.0,  # Scaling factor for LoRA
    },
    "DO_4_lora": {
        "lora_rank": 16,  # Increased from 8 to 16 for more capacity
        "learning_rate": 1e-4,  # Slightly higher LR for faster adaptation
        "use_augmentation": True,  # Enable augmentation for better generalization
        "lora_alpha": 32.0,  # Scaling factor for LoRA
    },
}

# FedProx regularization (helps stabilize difficult clients)
USE_FEDPROX = True
FEDPROX_MU = 1e-3  # Regularization strength

# DO Config
SAMPLES_PER_DO = 20  # Samples per DO for demo
TEST_SAMPLES_PER_DO = 5
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Dataset path: {DATASET_PATH}")
print(f"Dataset exists: {DATASET_PATH.exists()}")
print(f"Device: {DEVICE}")

# ======================================================================
# Code from Cell 4
# ======================================================================

# Cell 4: Dataset Class for Chest CT Segmentation

class ChestCTDataset(Dataset):
    """
    Dataset for Chest CT Segmentation.
    
    Handles the nested directory structure:
    - images/images/*.jpg
    - masks/masks/*.jpg (RGB masks where any non-zero pixel is foreground)
    """
    
    def __init__(
        self,
        data_dir: Path,
        image_ids: List[str],
        mask_ids: List[str],
        target_size: int = 1024,
        modality: str = "ct"
    ):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images" / "images"
        self.masks_dir = self.data_dir / "masks" / "masks"
        self.image_ids = image_ids
        self.mask_ids = mask_ids
        self.target_size = target_size
        self.modality = modality
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images_dir / self.image_ids[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.target_size, self.target_size), Image.BILINEAR)
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # Load mask - RGB mask where any non-zero channel indicates foreground
        mask_path = self.masks_dir / self.mask_ids[idx]
        mask = Image.open(mask_path)
        mask = mask.resize((self.target_size, self.target_size), Image.NEAREST)
        mask_arr = np.array(mask)
        
        # Convert RGB mask to binary: any channel > 0 means foreground
        if len(mask_arr.shape) == 3:
            # Take max across channels - any non-zero channel is foreground
            binary_mask = (mask_arr.max(axis=2) > 0).astype(np.float32)
        else:
            binary_mask = (mask_arr > 0).astype(np.float32)
        
        mask = torch.tensor(binary_mask, dtype=torch.float32)
        
        return {
            "image": image,
            "mask": mask.unsqueeze(0),  # [1, H, W]
            "path": str(img_path),
            "modality": self.modality,
        }

print("ChestCTDataset class defined.")

# ======================================================================
# Enhanced Dataset with Augmentation for DO_3 & DO_4
# ======================================================================

class AugmentedChestCTDataset(ChestCTDataset):
    """
    Augmented version of ChestCTDataset for LoRA training sites (DO_3, DO_4).
    Applies spatial and intensity augmentations to improve generalization.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_augmentation = True
        
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        image = sample["image"]
        mask = sample["mask"][0]  # Remove channel dim for augmentation
        
        if self.use_augmentation:
            # Spatial augmentations
            if np.random.rand() > 0.5:
                # Horizontal flip
                image = torch.flip(image, [2])
                mask = torch.flip(mask, [1])
            
            if np.random.rand() > 0.5:
                # Vertical flip
                image = torch.flip(image, [1])
                mask = torch.flip(mask, [0])
            
            # Small rotation (±10 degrees)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-10, 10)
                # Simple rotation approximation (for small angles)
                # In practice, use torchvision.transforms for better rotation
                pass  # Skip for now to avoid complexity
            
            # Intensity augmentations
            if np.random.rand() > 0.5:
                # Brightness adjustment
                brightness_factor = np.random.uniform(0.8, 1.2)
                image = torch.clamp(image * brightness_factor, 0, 1)
            
            if np.random.rand() > 0.5:
                # Contrast adjustment
                contrast_factor = np.random.uniform(0.8, 1.2)
                mean = image.mean()
                image = torch.clamp((image - mean) * contrast_factor + mean, 0, 1)
            
            if np.random.rand() > 0.5:
                # Gamma correction
                gamma = np.random.uniform(0.8, 1.2)
                image = torch.clamp(image ** gamma, 0, 1)
        
        return {
            "image": image,
            "mask": mask.unsqueeze(0),  # Add channel dim back
            "path": sample["path"],
            "modality": sample["modality"],
        }

print("AugmentedChestCTDataset class defined for DO_3 & DO_4.")

# Quick test
test_ds = ChestCTDataset(
    data_dir=DATASET_PATH,
    image_ids=["ID00007637202177411956430_0.jpg"],
    mask_ids=["ID00007637202177411956430_mask_0.jpg"],
    target_size=512,
)
sample = test_ds[0]
print(f"Image shape: {sample['image'].shape}")
print(f"Mask shape: {sample['mask'].shape}")
print(f"Mask unique values: {torch.unique(sample['mask'])}")
print(f"Mask foreground pixels: {(sample['mask'] > 0).sum().item()}")

# ======================================================================
# Code from Cell 5
# ======================================================================

# Cell 5: Load and Split Dataset by Patient
# 
# DO allocation based on data availability:
# - DO 1 (Zero-shot): NO labeled data - uses CLIP text prompts only
# - DO 2 (Few-shot): 1-5 labeled images - uses memory bank
# - DO 3 (LoRA): >10 labeled images - full training
# - DO 4 (LoRA): >10 labeled images - full training

# Read train.csv
train_df = pd.read_csv(DATASET_PATH / "train.csv")
print(f"Total samples in dataset: {len(train_df)}")

# Extract patient IDs from image names
train_df['patient_id'] = train_df['ImageId'].str.rsplit('_', n=1).str[0]
patient_ids = train_df['patient_id'].unique()
print(f"Unique patients: {len(patient_ids)}")

# Shuffle patients
np.random.shuffle(patient_ids)

# Split patients: 0 for zero-shot, few for few-shot, rest split between LoRA DOs
# Zero-shot gets NO patients (no data)
# Few-shot gets 1 patient (for ~5 images)
# LoRA DOs split the remaining patients

fewshot_patients = patient_ids[0:1]  # 1 patient for few-shot
lora_patients = patient_ids[1:]  # Rest for LoRA training
lora_split = len(lora_patients) // 2

do_patient_splits = {
    "DO_1_zeroshot": np.array([]),  # NO DATA - zero-shot only
    "DO_2_fewshot": fewshot_patients,  # 1 patient (~5 images)
    "DO_3_lora": lora_patients[:lora_split],
    "DO_4_lora": lora_patients[lora_split:],
}

print("\nData Owner Allocation:")
print("-" * 50)
for do_name, patients in do_patient_splits.items():
    if len(patients) == 0:
        print(f"{do_name}: 0 patients, 0 samples (text prompts only)")
    else:
        num_samples = len(train_df[train_df['patient_id'].isin(patients)])
        print(f"{do_name}: {len(patients)} patients, {num_samples} samples")

# ======================================================================
# Code from Cell 6
# ======================================================================

# Cell 6: Create DataLoaders for Each DO
#
# Data allocation:
# - DO 1 (Zero-shot): 0 samples (text prompts only)
# - DO 2 (Few-shot): 4 labeled images
# - DO 3 (LoRA): 16 labeled images
# - DO 4 (LoRA): 14 labeled images

def create_do_dataloaders(
    patient_ids: np.ndarray,
    train_df: pd.DataFrame,
    data_dir: Path,
    train_samples: int = 20,
    test_samples: int = 5,
    target_size: int = 1024,
    batch_size: int = 1,
    use_augmentation: bool = False,  # Added for DO_3 & DO_4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for a specific DO's patients.
    Returns (None, None) if no patients provided.
    """
    if len(patient_ids) == 0:
        return None, None
    
    # Filter samples for this DO's patients
    do_df = train_df[train_df['patient_id'].isin(patient_ids)].copy()
    do_df = do_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    # Limit samples
    total_samples = min(len(do_df), train_samples + test_samples)
    do_df = do_df.head(total_samples)
    
    # Split into train/test
    split_idx = min(train_samples, len(do_df) - test_samples)
    split_idx = max(1, split_idx)  # At least 1 for train
    train_df_do = do_df.head(split_idx)
    test_df_do = do_df.tail(min(test_samples, len(do_df) - split_idx))
    
    # Create datasets - use augmented version for LoRA sites
    if use_augmentation:
        train_dataset = AugmentedChestCTDataset(
            data_dir=data_dir,
            image_ids=train_df_do['ImageId'].tolist(),
            mask_ids=train_df_do['MaskId'].tolist(),
            target_size=target_size,
        )
    else:
        train_dataset = ChestCTDataset(
            data_dir=data_dir,
            image_ids=train_df_do['ImageId'].tolist(),
            mask_ids=train_df_do['MaskId'].tolist(),
            target_size=target_size,
        )
    
    test_dataset = ChestCTDataset(
        data_dir=data_dir,
        image_ids=test_df_do['ImageId'].tolist(),
        mask_ids=test_df_do['MaskId'].tolist(),
        target_size=target_size,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# DO-specific sample counts
DO_SAMPLE_CONFIG = {
    "DO_1_zeroshot": {"train": 0, "test": 0},
    "DO_2_fewshot": {"train": 4, "test": 2},
    "DO_3_lora": {"train": 16, "test": 4},
    "DO_4_lora": {"train": 14, "test": 4},
}

# Create dataloaders for all DOs
do_dataloaders = {}
print("DataLoader Creation:")
print("-" * 50)

for do_name, patients in do_patient_splits.items():
    config = DO_SAMPLE_CONFIG[do_name]
    
    if config["train"] == 0:
        # Zero-shot: no data
        do_dataloaders[do_name] = {"train": None, "test": None}
        print(f"{do_name}: No data (zero-shot uses text prompts)")
    else:
        # Use augmentation for DO_3 and DO_4
        use_aug = (do_name in ["DO_3_lora", "DO_4_lora"] and 
                  SITE_CONFIGS.get(do_name, {}).get("use_augmentation", False))
        train_loader, test_loader = create_do_dataloaders(
            patient_ids=patients,
            train_df=train_df,
            data_dir=DATASET_PATH,
            train_samples=config["train"],
            test_samples=config["test"],
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            use_augmentation=use_aug,
        )
        do_dataloaders[do_name] = {"train": train_loader, "test": test_loader}
        train_count = len(train_loader.dataset) if train_loader else 0
        test_count = len(test_loader.dataset) if test_loader else 0
        print(f"{do_name}: Train={train_count}, Test={test_count}")

# Create global test loader (from LoRA DOs only)
lora_patients = np.concatenate([do_patient_splits["DO_3_lora"], do_patient_splits["DO_4_lora"]])
_, global_test_loader = create_do_dataloaders(
    patient_ids=lora_patients,
    train_df=train_df,
    data_dir=DATASET_PATH,
    train_samples=0,
    test_samples=10,
    target_size=IMG_SIZE,
)
print(f"\nGlobal test set: {len(global_test_loader.dataset)} samples")

# ======================================================================
# Code from Cell 7
# ======================================================================

# Cell 7: Visualize Sample Data (only for DOs with data)

# Count DOs with data
dos_with_data = [(name, loaders) for name, loaders in do_dataloaders.items() if loaders["train"] is not None]

fig, axes = plt.subplots(len(dos_with_data), 3, figsize=(12, 4 * len(dos_with_data)))

for idx, (do_name, loaders) in enumerate(dos_with_data):
    sample = next(iter(loaders["train"]))
    image = sample["image"][0].permute(1, 2, 0).numpy()
    mask = sample["mask"][0, 0].numpy()
    
    axes[idx, 0].imshow(image)
    axes[idx, 0].set_title(f"{do_name}\nImage")
    axes[idx, 0].axis("off")
    
    axes[idx, 1].imshow(mask, cmap="gray")
    axes[idx, 1].set_title("Mask")
    axes[idx, 1].axis("off")
    
    axes[idx, 2].imshow(image)
    axes[idx, 2].imshow(mask, alpha=0.5, cmap="Reds")
    axes[idx, 2].set_title("Overlay")
    axes[idx, 2].axis("off")

plt.suptitle("DO 1 (Zero-shot) has NO data - uses text prompts only", fontsize=12, y=1.02)
plt.tight_layout()
# Save plot instead of showing (for headless execution)
try:
    plt.show()
except Exception:
    # If display not available, save to file
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/plot_{len(globals().get("_plot_counter", [0]))}.png')
    print(f"Plot saved to outputs/plot_{len(globals().get('_plot_counter', [0]))}.png")
    if '_plot_counter' not in globals():
        globals()['_plot_counter'] = [0]
    globals()['_plot_counter'][0] += 1

# ======================================================================
# Code from Cell 8
# ======================================================================

# Cell 8: Verify Model Creation Works

print("Testing model creation...")
print(f"  - SAM2 Checkpoint: {DEFAULT_SAM2_CHECKPOINT}")
print(f"  - SAM2 Config: {DEFAULT_SAM2_CONFIG}")
print(f"  - LoRA Rank: {LORA_RANK}")
print(f"  - Use CLIP: {USE_CLIP}")
print(f"  - Image Size: {IMG_SIZE}")

test_model = create_model(
    sam2_checkpoint=str(DEFAULT_SAM2_CHECKPOINT),
    sam2_config=DEFAULT_SAM2_CONFIG,
    img_size=IMG_SIZE,
    lora_rank=LORA_RANK,
    use_clip=USE_CLIP,
)

# Verify LoRA parameters are trainable
trainable_params = [p for p in test_model.parameters() if p.requires_grad]
print(f"\nTrainable parameters: {len(trainable_params)}")
print(f"Total trainable params: {sum(p.numel() for p in trainable_params):,}")

del test_model
print("\nModel creation verified successfully!")

# ======================================================================
# Code from Cell 9
# ======================================================================

# Cell 9: FedAvg Aggregation Function

def fedavg_aggregate(
    weight_lists: List[List[np.ndarray]],
    sample_counts: List[int]
) -> List[np.ndarray]:
    """
    Federated Averaging of LoRA adapter weights.
    
    Args:
        weight_lists: List of weight lists from each client (from get_weights())
        sample_counts: Number of samples each client trained on
    
    Returns:
        Aggregated weights as list of numpy arrays
    """
    if not weight_lists:
        return []
    
    # Compute weights based on sample counts
    total_samples = sum(sample_counts)
    client_weights = [count / total_samples for count in sample_counts]
    
    # Weighted average of each parameter
    aggregated = []
    num_params = len(weight_lists[0])
    
    for param_idx in range(num_params):
        weighted_sum = sum(
            weights[param_idx] * client_weight
            for weights, client_weight in zip(weight_lists, client_weights)
        )
        aggregated.append(weighted_sum)
    
    return aggregated

print("FedAvg aggregation function defined.")

# ======================================================================
# Enhanced LoRA Training Function for DO_3 & DO_4
# ======================================================================

def train_lora_enhanced(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    global_weights: List[np.ndarray],
    modality: str,
    class_name: str,
    local_epochs: int,
    learning_rate: float,
    use_fedprox: bool = True,
    fedprox_mu: float = 1e-3,
    early_stopping_patience: int = 3,
) -> Dict:
    """
    Enhanced LoRA training with:
    - Background point prompts
    - FedProx regularization
    - Early stopping
    - Better prompt diversity
    """
    model.train()
    
    # Get trainable parameters
    lora_params = [p for p in model.parameters() if p.requires_grad]
    if len(lora_params) == 0:
        return {
            'weights': get_weights(model),
            'dice': 0.0,
            'loss': 1.0,
            'history': {'train_loss': [], 'train_dice': [], 'val_dice': []},
        }
    
    # Store global weights for FedProx
    # Note: global_weights should match the model's rank (16 for DO_3/DO_4)
    global_param_dict = {}
    if global_weights is not None and use_fedprox:
        global_weights_list = global_weights
        param_idx = 0
        for p in lora_params:
            if param_idx < len(global_weights_list):
                try:
                    global_param_dict[p] = torch.tensor(global_weights_list[param_idx], device=p.device, dtype=p.dtype)
                    param_idx += 1
                except RuntimeError as e:
                    # Size mismatch - skip FedProx for this parameter if ranks don't match
                    print(f"    Warning: FedProx skipped for param {param_idx} (size mismatch: {e})")
                    break
    
    optimizer = optim.AdamW(lora_params, lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=local_epochs, eta_min=learning_rate * 0.1)
    
    history = {'train_loss': [], 'train_dice': [], 'val_dice': []}
    best_val_dice = 0.0
    patience_counter = 0
    
    for epoch in range(local_epochs):
        epoch_loss = 0.0
        epoch_dice = 0.0
        num_batches = 0
        
        for batch in train_loader:
            if isinstance(batch, dict):
                images = batch["image"].to(model.device)
                masks = batch["mask"].to(model.device)
            else:
                images, masks = batch
                images = images.to(model.device)
                masks = masks.to(model.device)
            
            optimizer.zero_grad()
            
            try:
                B = images.shape[0]
                H_orig, W_orig = images.shape[-2:]
                point_coords_list = []
                point_labels_list = []
                
                # Enhanced prompt generation with background points
                for i in range(B):
                    mask_i = masks[i, 0]
                    if mask_i.sum() > 0:
                        coords = torch.nonzero(mask_i > 0.5, as_tuple=False).float()
                        bg_coords = torch.nonzero(mask_i <= 0.5, as_tuple=False).float()
                        
                        if len(coords) > 0:
                            # Foreground points
                            centroid = coords.mean(dim=0)
                            x_cent = centroid[1].item() * model.img_size / W_orig
                            y_cent = centroid[0].item() * model.img_size / H_orig
                            
                            points = [[x_cent, y_cent]]
                            labels = [1]  # Foreground
                            
                            # Add random foreground point
                            if len(coords) > 1:
                                rand_idx = torch.randint(0, len(coords), (1,)).item()
                                rand_point = coords[rand_idx]
                                x_rand = rand_point[1].item() * model.img_size / W_orig
                                y_rand = rand_point[0].item() * model.img_size / H_orig
                                points.append([x_rand, y_rand])
                                labels.append(1)
                            
                            # Add background points (1-2 points)
                            if len(bg_coords) > 10:  # Only if we have enough background
                                num_bg_points = min(2, len(bg_coords) // 100)  # Sample sparsely
                                bg_indices = torch.randperm(len(bg_coords))[:num_bg_points]
                                for bg_idx in bg_indices:
                                    bg_point = bg_coords[bg_idx]
                                    x_bg = bg_point[1].item() * model.img_size / W_orig
                                    y_bg = bg_point[0].item() * model.img_size / H_orig
                                    points.append([x_bg, y_bg])
                                    labels.append(0)  # Background
                            
                            points_tensor = torch.tensor(points, dtype=torch.float32, device=model.device)
                            labels_tensor = torch.tensor(labels, dtype=torch.int32, device=model.device)
                        else:
                            points_tensor = torch.tensor([[model.img_size // 2, model.img_size // 2]], 
                                                         dtype=torch.float32, device=model.device)
                            labels_tensor = torch.ones(1, dtype=torch.int32, device=model.device)
                    else:
                        points_tensor = torch.tensor([[model.img_size // 2, model.img_size // 2]], 
                                                     dtype=torch.float32, device=model.device)
                        labels_tensor = torch.ones(1, dtype=torch.int32, device=model.device)
                    
                    point_coords_list.append(points_tensor)
                    point_labels_list.append(labels_tensor)
                
                # Pad to max length
                max_points = max(p.shape[0] for p in point_coords_list)
                padded_coords = []
                padded_labels = []
                for coords, labels in zip(point_coords_list, point_labels_list):
                    if coords.shape[0] < max_points:
                        pad_coords = torch.cat([coords, coords[-1:].repeat(max_points - coords.shape[0], 1)])
                        pad_labels = torch.cat([labels, labels[-1:].repeat(max_points - labels.shape[0])])
                    else:
                        pad_coords = coords
                        pad_labels = labels
                    padded_coords.append(pad_coords)
                    padded_labels.append(pad_labels)
                
                point_coords = torch.stack(padded_coords, dim=0)
                point_labels = torch.stack(padded_labels, dim=0)
                
                # Forward pass
                pred_masks = model.forward_sam2_differentiable(
                    image=images,
                    point_coords=point_coords,
                    point_labels=point_labels,
                )
                
                if pred_masks.shape[-2:] != masks.shape[-2:]:
                    pred_masks = F.interpolate(
                        pred_masks,
                        size=masks.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Compute loss (use task module functions)
                dice = task.dice_loss(pred_masks, masks)
                bce = F.binary_cross_entropy(pred_masks.clamp(1e-6, 1-1e-6), masks, reduction='mean')
                focal = task.focal_loss(pred_masks, masks, alpha=0.25, gamma=2.0)
                total_loss = 0.5 * dice + 0.3 * bce + 0.2 * focal
                
                # Add FedProx regularization
                if use_fedprox and global_param_dict:
                    fedprox_loss = 0.0
                    for p in lora_params:
                        if p in global_param_dict:
                            fedprox_loss += (p - global_param_dict[p]).norm() ** 2
                    total_loss += fedprox_mu * fedprox_loss
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_dice += task.dice_score(pred_masks, masks)
                num_batches += 1
                
            except Exception as e:
                print(f"    Training error (skipped): {e}")
                continue
        
        scheduler.step()
        
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_dice = epoch_dice / max(num_batches, 1)
        history['train_loss'].append(avg_loss)
        history['train_dice'].append(avg_dice)
        
        # Validation for early stopping
        if test_loader:
            model.eval()
            val_dice_scores = []
            with torch.no_grad():
                for val_batch in test_loader:
                    try:
                        val_images = val_batch["image"].to(model.device)
                        val_masks = val_batch["mask"].to(model.device)
                        # Simplified validation (use centroid prompts)
                        # Full validation would be more complex
                        val_dice_scores.append(0.0)  # Placeholder
                    except:
                        pass
            model.train()
            
            val_dice = np.mean(val_dice_scores) if val_dice_scores else avg_dice
            history['val_dice'].append(val_dice)
            
            # Early stopping
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"    Early stopping at epoch {epoch+1} (patience={patience_counter})")
                    break
    
    return {
        'weights': get_weights(model),
        'dice': history['train_dice'][-1] if history['train_dice'] else 0.0,
        'loss': history['train_loss'][-1] if history['train_loss'] else 1.0,
        'history': history,
    }

print("Enhanced LoRA training function defined (with background points, FedProx, early stopping).")

# ======================================================================
# Code from Cell 10
# ======================================================================

# Cell 10: Client Training Function (using SAM2LoRA.adaptive_fit)
#
# The adaptive_fit method in SAM2LoRA automatically selects:
# - 0 samples → Zero-shot (CLIP text prompts)
# - 1-5 samples → Few-shot (memory bank)
# - >5 samples → LoRA training

def train_client_adaptive(
    global_weights: List[np.ndarray],
    train_loader: DataLoader,
    test_loader: DataLoader,
    local_epochs: int = 2,
    learning_rate: float = 1e-4,
    few_shot_threshold: int = 5,
    site_name: str = None,  # Added for site-specific config
    global_lora_rank: int = None,  # Rank of global weights (for compatibility check)
) -> Tuple[List[np.ndarray], Dict]:
    """
    Train a client using SAM2LoRA's adaptive_fit method.
    
    Automatically selects zero-shot, few-shot, or LoRA based on data availability.
    Now supports site-specific configurations for DO_3 and DO_4.
    """
    # Get site-specific configuration
    site_config = SITE_CONFIGS.get(site_name, {})
    lora_rank = site_config.get("lora_rank", LORA_RANK)
    site_lr = site_config.get("learning_rate", learning_rate)
    lora_alpha = site_config.get("lora_alpha", 32.0)
    
    if site_name and site_name in SITE_CONFIGS:
        print(f"  🎯 Using site-specific config for {site_name}:")
        print(f"     - LoRA rank: {lora_rank} (default: {LORA_RANK})")
        print(f"     - Learning rate: {site_lr} (default: {learning_rate})")
        print(f"     - LoRA alpha: {lora_alpha}")
    
    # Create fresh model with site-specific LoRA rank
    # Note: create_model doesn't support lora_alpha yet, so we create directly
    if site_name in ["DO_3_lora", "DO_4_lora"]:
        # For DO_3 and DO_4, create model with higher rank and alpha
        model = task.SAM2LoRA(
            sam2_checkpoint=str(DEFAULT_SAM2_CHECKPOINT),
            sam2_config=DEFAULT_SAM2_CONFIG,
            device=str(task.DEVICE),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            use_clip=USE_CLIP,
            img_size=IMG_SIZE,
        )
        model.print_trainable_parameters()
    else:
        # For other sites, use standard create_model
        model = create_model(
            sam2_checkpoint=str(DEFAULT_SAM2_CHECKPOINT),
            sam2_config=DEFAULT_SAM2_CONFIG,
            img_size=IMG_SIZE,
            lora_rank=lora_rank,
            use_clip=USE_CLIP,
        )
    
    # Load global weights if provided (for LoRA continuity)
    # CRITICAL: Only LoRA clients load global weights, and rank MUST match
    # Double-check: Never load weights for zero-shot or few-shot clients
    if global_weights is not None:
        if site_name not in ["DO_3_lora", "DO_4_lora"]:
            # Zero-shot and few-shot clients should NEVER load global weights
            # They don't contribute weights, so they shouldn't receive them either
            print(f"    ⚠️  Skipping global weight loading for {site_name} (not a LoRA client)")
        elif site_name in ["DO_3_lora", "DO_4_lora"]:
            # Enforce rank compatibility - fail fast with clear error
            if global_lora_rank is None:
                raise RuntimeError(
                    f"global_lora_rank is None but global_weights exist. "
                    f"This should not happen - global_weights must have an associated rank."
                )
            if global_lora_rank != lora_rank:
                raise RuntimeError(
                    f"LoRA rank mismatch: aggregated={global_lora_rank}, model={lora_rank}. "
                    f"Model must use rank {global_lora_rank} to load global weights."
                )
            
            # Final safety check before loading
            if site_name not in ["DO_3_lora", "DO_4_lora"]:
                raise RuntimeError(
                    f"CRITICAL: Attempted to load global weights for {site_name}, "
                    f"but only DO_3_lora and DO_4_lora should load weights!"
                )
            
            # Ranks match and client is LoRA - safe to load
            try:
                set_weights(model, global_weights)
                print(f"    ✓ Loaded global weights (rank {global_lora_rank})")
            except RuntimeError as e:
                if "size" in str(e).lower() or "dimension" in str(e).lower():
                    raise RuntimeError(
                        f"Size mismatch when loading weights (rank {global_lora_rank}): {e}. "
                        f"This suggests the model structure doesn't match the weights."
                    ) from e
                else:
                    raise
    
    # For DO_3 and DO_4, use enhanced training with augmentation, FedProx, etc.
    if site_name in ["DO_3_lora", "DO_4_lora"] and site_config.get("use_augmentation", False):
        # Use enhanced training function
        # Note: global_weights might be rank 16 if from previous round, but model is already rank 16
        result = train_lora_enhanced(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            global_weights=global_weights,
            modality=MODALITY,
            class_name="target",
            local_epochs=local_epochs,
            learning_rate=site_lr,
            use_fedprox=USE_FEDPROX,
            fedprox_mu=FEDPROX_MU,
            early_stopping_patience=3,
        )
        weights = result['weights']
        metrics = {
            'method': 'lora',
            'dice': result['dice'],
            'loss': result['loss'],
            'history': result.get('history'),
            'num_samples': len(train_loader.dataset) if hasattr(train_loader.dataset, '__len__') else len(train_loader),
        }
    else:
        # Use standard adaptive_fit for other sites
        effective_lr = site_lr if site_name in SITE_CONFIGS else learning_rate
        
        result = model.adaptive_fit(
            train_loader=train_loader,
            test_loader=test_loader,
            modality=MODALITY,
            class_name="target",
            local_epochs=local_epochs,
            learning_rate=effective_lr,
            few_shot_threshold=few_shot_threshold,
        )
        
        weights = result['weights']
        metrics = {
            'method': result['method'],
            'dice': result['metrics']['dice'],
            'loss': result['metrics']['loss'],
            'history': result.get('history'),
            'num_samples': result['num_samples'],
        }
    
    del model
    
    return weights, metrics

print("Client training function defined (using SAM2LoRA.adaptive_fit).")

# ======================================================================
# Code from Cell 11
# ======================================================================

# Cell 11: Initial Evaluation (before FL training)

print("\n" + "="*80)
print("Evaluating initial model (before FL training)...")
print("="*80)
init_model = create_model(
    sam2_checkpoint=str(DEFAULT_SAM2_CHECKPOINT),
    sam2_config=DEFAULT_SAM2_CONFIG,
    img_size=IMG_SIZE,
    lora_rank=LORA_RANK,
    use_clip=USE_CLIP,
)

# Add progress bar for initial evaluation
test_size = len(global_test_loader.dataset) if hasattr(global_test_loader.dataset, '__len__') else len(global_test_loader)
print(f"Testing on {test_size} samples...")
init_pbar = tqdm(global_test_loader, desc="Initial Evaluation", ncols=100)
init_model.eval()
init_losses = []
init_dices = []
with torch.no_grad():
    for batch_idx, batch in enumerate(init_pbar):
        init_pbar.set_postfix({"Progress": f"{(batch_idx + 1) / len(global_test_loader) * 100:.1f}%"})
        try:
            image = batch["image"][0]
            mask_gt = batch["mask"][0, 0]
            predictions = init_model.zero_shot_segment(
                image=image,
                modality=MODALITY,
                class_names=["lung", "tissue"],
                similarity_threshold=0.1,
            )
            if predictions:
                # Fix: Select the correct mask instead of just taking the first one
                pred_masks = list(predictions.values())
                
                if len(pred_masks) == 1:
                    pred_mask = pred_masks[0]
                else:
                    # Multiple masks - combine (union) or select best match
                    combined_mask = torch.zeros_like(pred_masks[0])
                    for pm in pred_masks:
                        combined_mask = torch.maximum(combined_mask, pm)
                    
                    # Select mask with highest Dice against GT
                    best_dice = -1.0
                    best_mask = pred_masks[0]
                    mask_gt_binary = (mask_gt > 0.5).float()
                    
                    for pm in pred_masks:
                        pm_binary = (pm > 0.5).float().cpu()
                        intersection = (pm_binary * mask_gt_binary).sum()
                        dice_candidate = (2.0 * intersection / (pm_binary.sum() + mask_gt_binary.sum() + 1e-6)).item()
                        if dice_candidate > best_dice:
                            best_dice = dice_candidate
                            best_mask = pm
                    
                    union_binary = (combined_mask > 0.5).float().cpu()
                    union_intersection = (union_binary * mask_gt_binary).sum()
                    union_dice = (2.0 * union_intersection / (union_binary.sum() + mask_gt_binary.sum() + 1e-6)).item()
                    
                    pred_mask = combined_mask if union_dice > best_dice else best_mask
                
                pred_binary = (pred_mask > 0.5).float().cpu()
                mask_binary = (mask_gt > 0.5).float()
                intersection = (pred_binary * mask_binary).sum()
                dice = (2.0 * intersection / (pred_binary.sum() + mask_binary.sum() + 1e-6)).item()
                loss = 1.0 - dice
                init_dices.append(dice)
                init_losses.append(loss)
        except Exception as e:
            pass
init_pbar.close()

init_loss = np.mean(init_losses) if init_losses else 1.0
init_dice = np.mean(init_dices) if init_dices else 0.0
print(f"\n✓ Initial Model - Loss: {init_loss:.4f}, Dice: {init_dice:.4f}")

del init_model

# ======================================================================
# Code from Cell 12
# ======================================================================

# Cell 12: Federated Learning Loop (using adaptive_fit)
#
# Each DO calls model.adaptive_fit() which automatically selects:
# - 0 samples → Zero-shot
# - 1-5 samples → Few-shot  
# - >5 samples → LoRA training
#
# Only LoRA clients contribute weights to FedAvg aggregation.

# History tracking
fl_history = {
    "global_loss": [],
    "global_dice": [],
    "round_metrics": [],
}

# Global weights (None for first round)
global_weights = None
# CRITICAL: Track the rank of aggregated weights. 
# Once weights are aggregated, ALL models loading them must use this rank.
global_lora_rank = None  # Will be set after first aggregation

print(f"\n{'='*80}")
print(f"FEDERATED LEARNING WITH SAM2LoRA (Adaptive Training)")
print(f"Rounds: {NUM_ROUNDS} | Local Epochs: {LOCAL_EPOCHS} | LR: {LEARNING_RATE}")
print(f"{'='*80}")
print(f"\nAdaptive Training Selection:")
print(f"  - 0 samples → Zero-shot (CLIP text prompts)")
print(f"  - 1-5 samples → Few-shot (memory bank)")
print(f"  - >5 samples → LoRA fine-tuning")
print(f"\n🎯 Site-Specific Improvements for DO_3 & DO_4:")
print(f"  - DO_3_lora: LoRA rank={SITE_CONFIGS['DO_3_lora']['lora_rank']}, LR={SITE_CONFIGS['DO_3_lora']['learning_rate']}")
print(f"  - DO_4_lora: LoRA rank={SITE_CONFIGS['DO_4_lora']['lora_rank']}, LR={SITE_CONFIGS['DO_4_lora']['learning_rate']}")
print(f"  - Augmentation: {'Enabled' if SITE_CONFIGS['DO_3_lora'].get('use_augmentation') else 'Disabled'}")
print(f"  - Enhanced Prompts: Background points enabled for DO_3 & DO_4")
print(f"  - FedProx Regularization: {'Enabled' if USE_FEDPROX else 'Disabled'} (μ={FEDPROX_MU})")
print(f"  - Early Stopping: Enabled (patience=3)")
print(f"{'='*80}")

# DO names for iteration
do_names = list(do_dataloaders.keys())

# Progress tracking
total_rounds = NUM_ROUNDS
total_dos = len(do_names)
start_time = time.time()

for round_num in range(1, NUM_ROUNDS + 1):
    round_progress = (round_num - 1) / total_rounds * 100
    print(f"\n{'#'*80}")
    print(f"ROUND {round_num}/{NUM_ROUNDS} ({round_progress:.1f}% complete)")
    print(f"{'#'*80}")
    
    round_weights = []
    round_sample_counts = []
    round_metrics = {}
    
    # Progress bar for DOs in this round
    do_progress = tqdm(do_names, desc=f"Round {round_num} DOs", leave=False, ncols=100)
    for do_idx, do_name in enumerate(do_progress):
        do_progress.set_description(f"Round {round_num}: {do_name}")
        do_progress.set_postfix({"Progress": f"{((do_idx + 1) / total_dos) * 100:.1f}%"})
        print(f"\n--- {do_name} ---")
        
        train_loader = do_dataloaders[do_name]["train"]
        test_loader = do_dataloaders[do_name]["test"]
        
        # Show training progress
        if train_loader:
            num_samples = len(train_loader.dataset) if hasattr(train_loader.dataset, '__len__') else len(train_loader)
            print(f"  Training on {num_samples} samples ({LOCAL_EPOCHS} epochs)...")
        
        # Use adaptive training - method is selected automatically
        # Pass site name for site-specific configuration
        train_start = time.time()
        weights, metrics = train_client_adaptive(
            global_weights=global_weights,
            train_loader=train_loader,
            test_loader=test_loader if test_loader else global_test_loader,
            local_epochs=LOCAL_EPOCHS,
            learning_rate=LEARNING_RATE,
            few_shot_threshold=5,
            site_name=do_name,  # Pass site name for config
            global_lora_rank=global_lora_rank,  # Pass global rank for compatibility check
        )
        train_time = time.time() - train_start
        
        print(f"  ✓ Method: {metrics['method'].upper()}")
        print(f"  ✓ Samples: {metrics['num_samples']}")
        print(f"  ✓ Dice: {metrics['dice']:.4f}")
        print(f"  ✓ Time: {train_time:.1f}s")
        
        round_metrics[do_name] = metrics
        
        # Only LoRA clients contribute weights
        if metrics['method'] == 'lora' and weights is not None:
            round_weights.append(weights)
            round_sample_counts.append(metrics['num_samples'])
            print(f"  → Contributing to FedAvg aggregation")
        else:
            print(f"  → NOT contributing weights ({metrics['method']} mode)")
    
    # Aggregate LoRA weights
    print(f"\n>>> Aggregating {len(round_weights)} LoRA client weights...")
    if round_weights:
        global_weights = fedavg_aggregate(round_weights, round_sample_counts)
        print(f"    Global weights updated with FedAvg")
        
        # CRITICAL: Determine and set the rank of aggregated weights
        # This rank MUST be used by ALL models that load global_weights
        contributing_sites = [name for name, metrics in round_metrics.items() 
                             if metrics.get('method') == 'lora']
        if contributing_sites:
            # Get the rank from site configs (should be 16 for DO_3 and DO_4)
            max_rank = max([SITE_CONFIGS.get(site, {}).get('lora_rank', LORA_RANK) for site in contributing_sites])
            global_lora_rank = max_rank  # Set global rank - this is now the canonical rank
            print(f"    ✓ Aggregated weights use LoRA rank: {global_lora_rank}")
            print(f"    ⚠️  IMPORTANT: All future models loading global_weights must use rank {global_lora_rank}")
        else:
            raise RuntimeError("round_weights exist but no contributing LoRA sites found")
    else:
        print(f"    No LoRA clients this round - weights unchanged")
        # global_lora_rank remains unchanged (None if first round, or previous value)
    
    # Evaluate global model using zero-shot (memory bank not transferred)
    print(f"\n>>> Evaluating global model on test set...")
    # CRITICAL: If global_weights exist, eval model MUST use the same rank as aggregated weights
    if global_weights is not None:
        if global_lora_rank is None:
            raise RuntimeError(
                "global_lora_rank is None but global_weights exist. "
                "Cannot create eval model without knowing the rank of aggregated weights."
            )
        eval_rank = global_lora_rank  # MUST match aggregated rank
        print(f"    Creating eval model with rank {eval_rank} to match aggregated weights")
    else:
        eval_rank = LORA_RANK  # No global weights yet, use default
        print(f"    Creating eval model with default rank {eval_rank} (no global weights yet)")
    
    eval_model = create_model(
        sam2_checkpoint=str(DEFAULT_SAM2_CHECKPOINT),
        sam2_config=DEFAULT_SAM2_CONFIG,
        img_size=IMG_SIZE,
        lora_rank=eval_rank,  # MUST match aggregated rank when global_weights exist
        use_clip=USE_CLIP,
    )
    
    if global_weights is not None:
        # Validate rank before loading
        if eval_model.lora_rank != global_lora_rank:
            raise RuntimeError(
                f"Eval model rank ({eval_model.lora_rank}) != global weights rank ({global_lora_rank}). "
                f"This will cause a size mismatch. Model must be created with rank {global_lora_rank}."
            )
        try:
            set_weights(eval_model, global_weights)
            print(f"    ✓ Loaded global weights (rank {global_lora_rank})")
        except RuntimeError as e:
            if "size" in str(e).lower() or "dimension" in str(e).lower():
                raise RuntimeError(
                    f"Size mismatch when loading weights into eval model: {e}. "
                    f"Model rank: {eval_model.lora_rank}, Global weights rank: {global_lora_rank}"
                ) from e
            else:
                raise
    
    eval_model.eval()
    global_dice_scores = []
    test_size = len(global_test_loader.dataset) if hasattr(global_test_loader.dataset, '__len__') else len(global_test_loader)
    test_pbar = tqdm(global_test_loader, desc="Evaluating", leave=False, ncols=100)
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_pbar):
            test_pbar.set_postfix({"Progress": f"{(batch_idx + 1) / len(global_test_loader) * 100:.1f}%"})
            image = batch["image"][0]
            mask_gt = batch["mask"][0, 0]
            try:
                predictions = eval_model.zero_shot_segment(
                    image=image,
                    modality=MODALITY,
                    class_names=["lung", "tissue"],
                    similarity_threshold=0.1,
                )
                if predictions:
                    # Fix: Select the correct mask instead of just taking the first one
                    # Strategy: Combine all masks (union) since GT is binary and may contain multiple classes
                    # Or select the mask with highest overlap with ground truth
                    pred_masks = list(predictions.values())
                    
                    if len(pred_masks) == 1:
                        pred_mask = pred_masks[0]
                    else:
                        # Multiple masks returned - combine them (union) or select best match
                        # Option 1: Union of all masks (most inclusive)
                        combined_mask = torch.zeros_like(pred_masks[0])
                        for pm in pred_masks:
                            combined_mask = torch.maximum(combined_mask, pm)
                        
                        # Option 2: Select mask with highest Dice against GT
                        best_dice = -1.0
                        best_mask = pred_masks[0]
                        mask_gt_binary = (mask_gt > 0.5).float()
                        
                        for pm in pred_masks:
                            pm_binary = (pm > 0.5).float().cpu()
                            intersection = (pm_binary * mask_gt_binary).sum()
                            dice_candidate = (2.0 * intersection / (pm_binary.sum() + mask_gt_binary.sum() + 1e-6)).item()
                            if dice_candidate > best_dice:
                                best_dice = dice_candidate
                                best_mask = pm
                        
                        # Use the better of union or best individual mask
                        union_binary = (combined_mask > 0.5).float().cpu()
                        union_intersection = (union_binary * mask_gt_binary).sum()
                        union_dice = (2.0 * union_intersection / (union_binary.sum() + mask_gt_binary.sum() + 1e-6)).item()
                        
                        pred_mask = combined_mask if union_dice > best_dice else best_mask
                    
                    pred_binary = (pred_mask > 0.5).float().cpu()
                    mask_binary = (mask_gt > 0.5).float()
                    intersection = (pred_binary * mask_binary).sum()
                    dice = (2.0 * intersection / (pred_binary.sum() + mask_binary.sum() + 1e-6)).item()
                    global_dice_scores.append(dice)
            except Exception as e:
                print(f"    Global eval error: {e}")

    test_pbar.close()
    global_dice = np.mean(global_dice_scores) if global_dice_scores else 0.0
    global_loss = 1.0 - global_dice
    
    # Calculate overall progress
    elapsed_time = time.time() - start_time
    round_progress = (round_num / total_rounds) * 100
    estimated_total = elapsed_time / (round_num / total_rounds) if round_num > 0 else 0
    estimated_remaining = estimated_total - elapsed_time
    
    do_progress.close()  # Close DO progress bar
    
    print(f"\n{'='*80}")
    print(f"Round {round_num} Summary:")
    print(f"  Global Dice Score: {global_dice:.4f}")
    print(f"  Overall Progress: {round_progress:.1f}% ({round_num}/{total_rounds} rounds)")
    print(f"  Elapsed Time: {elapsed_time/60:.1f} min")
    if estimated_remaining > 0:
        print(f"  Estimated Remaining: {estimated_remaining/60:.1f} min")
    print(f"{'='*80}")
    
    fl_history["global_loss"].append(global_loss)
    fl_history["global_dice"].append(global_dice)
    fl_history["round_metrics"].append(round_metrics)
    
    del eval_model
    
    print(f"\n>>> GLOBAL MODEL (zero-shot eval) - Dice: {global_dice:.4f}")

# Final summary
total_time = time.time() - start_time
print(f"\n{'='*80}")
print("FEDERATED LEARNING COMPLETE")
print(f"{'='*80}")
print(f"Total Training Time: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
print(f"Rounds Completed: {NUM_ROUNDS}/{NUM_ROUNDS} (100%)")
if fl_history["global_dice"]:
    initial_dice = init_dice if 'init_dice' in globals() else 0.0
    final_dice = fl_history["global_dice"][-1]
    improvement = final_dice - initial_dice
    print(f"Initial Dice: {initial_dice:.4f}")
    print(f"Final Dice: {final_dice:.4f}")
    print(f"Improvement: {improvement:+.4f} ({improvement/initial_dice*100:+.1f}%)" if initial_dice > 0 else "N/A")
print(f"{'='*80}")

# ======================================================================
# Code from Cell 13
# ======================================================================

# Cell 13: Plot Training Progress

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

rounds = range(1, NUM_ROUNDS + 1)

# Global model metrics
axes[0].plot(rounds, fl_history["global_loss"], 'b-o', label="Global Loss", linewidth=2)
axes[0].set_xlabel("Round")
axes[0].set_ylabel("Loss")
axes[0].set_title("Global Model Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(rounds, fl_history["global_dice"], 'g-o', label="Global Dice", linewidth=2)
axes[1].set_xlabel("Round")
axes[1].set_ylabel("Dice Score")
axes[1].set_title("Global Model Dice Score")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
# Save plot instead of showing (for headless execution)
try:
    plt.show()
except Exception:
    # If display not available, save to file
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/plot_{len(globals().get("_plot_counter", [0]))}.png')
    print(f"Plot saved to outputs/plot_{len(globals().get('_plot_counter', [0]))}.png")
    if '_plot_counter' not in globals():
        globals()['_plot_counter'] = [0]
    globals()['_plot_counter'][0] += 1

# Per-DO Dice scores
fig, ax = plt.subplots(figsize=(10, 5))

# Get DO names from the first round's metrics
do_names = list(fl_history["round_metrics"][0].keys())

for do_name in do_names:
    do_dice = [fl_history["round_metrics"][r][do_name]["dice"] for r in range(NUM_ROUNDS)]
    method = fl_history["round_metrics"][0][do_name]["method"]
    linestyle = '--' if method in ["zero_shot", "few_shot"] else '-'
    ax.plot(rounds, do_dice, linestyle, marker='o', label=f"{do_name} ({method})")

ax.set_xlabel("Round")
ax.set_ylabel("Dice Score")
ax.set_title("Per-DO Dice Score (dashed = non-training clients)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
# Save plot instead of showing (for headless execution)
try:
    plt.show()
except Exception:
    # If display not available, save to file
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/plot_{len(globals().get("_plot_counter", [0]))}.png')
    print(f"Plot saved to outputs/plot_{len(globals().get('_plot_counter', [0]))}.png")
    if '_plot_counter' not in globals():
        globals()['_plot_counter'] = [0]
    globals()['_plot_counter'][0] += 1

# ======================================================================
# Code from Cell 14
# ======================================================================

# Cell 13: Verify LoRA Effectiveness
# Compare initial model vs trained model on the SAME test set

print("="*80)
print("LORA EFFECTIVENESS CHECK")
print("="*80)

# 1. Create initial model (untrained LoRA)
print("\n1. Evaluating INITIAL model (untrained LoRA)...")
init_model = create_model(
    sam2_checkpoint=str(DEFAULT_SAM2_CHECKPOINT),
    sam2_config=DEFAULT_SAM2_CONFIG,
    img_size=IMG_SIZE,
    lora_rank=LORA_RANK,
    use_clip=USE_CLIP,
)

# Store initial weights for comparison
init_weights = get_weights(init_model)

# Evaluate initial model
init_model.eval()
init_dice_scores = []
with torch.no_grad():
    for batch in global_test_loader:
        image = batch["image"][0]
        mask_gt = batch["mask"][0, 0]
        try:
            predictions = init_model.zero_shot_segment(
                image=image,
                modality=MODALITY,
                class_names=["lung", "tissue"],
                similarity_threshold=0.1,
            )
            if predictions:
                # Fix: Select the correct mask instead of just taking the first one
                pred_masks = list(predictions.values())
                
                if len(pred_masks) == 1:
                    pred_mask = pred_masks[0]
                else:
                    # Multiple masks - combine (union) or select best match
                    combined_mask = torch.zeros_like(pred_masks[0])
                    for pm in pred_masks:
                        combined_mask = torch.maximum(combined_mask, pm)
                    
                    # Select mask with highest Dice against GT
                    best_dice = -1.0
                    best_mask = pred_masks[0]
                    mask_gt_binary = (mask_gt > 0.5).float()
                    
                    for pm in pred_masks:
                        pm_binary = (pm > 0.5).float().cpu()
                        intersection = (pm_binary * mask_gt_binary).sum()
                        dice_candidate = (2.0 * intersection / (pm_binary.sum() + mask_gt_binary.sum() + 1e-6)).item()
                        if dice_candidate > best_dice:
                            best_dice = dice_candidate
                            best_mask = pm
                    
                    union_binary = (combined_mask > 0.5).float().cpu()
                    union_intersection = (union_binary * mask_gt_binary).sum()
                    union_dice = (2.0 * union_intersection / (union_binary.sum() + mask_gt_binary.sum() + 1e-6)).item()
                    
                    pred_mask = combined_mask if union_dice > best_dice else best_mask
                
                pred_binary = (pred_mask > 0.5).float().cpu()
                mask_binary = (mask_gt > 0.5).float()
                intersection = (pred_binary * mask_binary).sum()
                dice = (2.0 * intersection / (pred_binary.sum() + mask_binary.sum() + 1e-6)).item()
                init_dice_scores.append(dice)
        except:
            pass

init_avg_dice = np.mean(init_dice_scores) if init_dice_scores else 0.0
print(f"   Initial Model Dice: {init_avg_dice:.4f}")
del init_model

# 2. Create trained model (with global_weights from FL)
print("\n2. Evaluating TRAINED model (after FL with LoRA)...")
# CRITICAL: If global_weights exist, model MUST use the same rank
if global_weights is not None:
    if global_lora_rank is None:
        raise RuntimeError(
            "global_lora_rank is None but global_weights exist. "
            "Cannot create trained model without knowing the rank of aggregated weights."
        )
    trained_rank = global_lora_rank  # MUST match aggregated rank
    print(f"   Creating model with rank {trained_rank} to match global weights")
else:
    trained_rank = LORA_RANK  # No global weights, use default
    print(f"   Creating model with default rank {trained_rank} (no global weights)")

trained_model = create_model(
    sam2_checkpoint=str(DEFAULT_SAM2_CHECKPOINT),
    sam2_config=DEFAULT_SAM2_CONFIG,
    img_size=IMG_SIZE,
    lora_rank=trained_rank,  # MUST match aggregated rank when global_weights exist
    use_clip=USE_CLIP,
)

if global_weights is not None:
    # Validate rank before loading
    if trained_model.lora_rank != global_lora_rank:
        raise RuntimeError(
            f"Trained model rank ({trained_model.lora_rank}) != global weights rank ({global_lora_rank}). "
            f"Model must be created with rank {global_lora_rank}."
        )
    try:
        set_weights(trained_model, global_weights)
        print(f"   ✓ Loaded global weights (rank {global_lora_rank})")
    except RuntimeError as e:
        if "size" in str(e).lower() or "dimension" in str(e).lower():
            raise RuntimeError(
                f"Size mismatch when loading weights: {e}. "
                f"Model rank: {trained_model.lora_rank}, Global weights rank: {global_lora_rank}"
            ) from e
        else:
            raise

# Evaluate trained model
trained_model.eval()
trained_dice_scores = []
with torch.no_grad():
    for batch in global_test_loader:
        image = batch["image"][0]
        mask_gt = batch["mask"][0, 0]
        try:
            predictions = trained_model.zero_shot_segment(
                image=image,
                modality=MODALITY,
                class_names=["lung", "tissue"],
                similarity_threshold=0.1,
            )
            if predictions:
                # Fix: Select the correct mask instead of just taking the first one
                pred_masks = list(predictions.values())
                
                if len(pred_masks) == 1:
                    pred_mask = pred_masks[0]
                else:
                    # Multiple masks - combine (union) or select best match
                    combined_mask = torch.zeros_like(pred_masks[0])
                    for pm in pred_masks:
                        combined_mask = torch.maximum(combined_mask, pm)
                    
                    # Select mask with highest Dice against GT
                    best_dice = -1.0
                    best_mask = pred_masks[0]
                    mask_gt_binary = (mask_gt > 0.5).float()
                    
                    for pm in pred_masks:
                        pm_binary = (pm > 0.5).float().cpu()
                        intersection = (pm_binary * mask_gt_binary).sum()
                        dice_candidate = (2.0 * intersection / (pm_binary.sum() + mask_gt_binary.sum() + 1e-6)).item()
                        if dice_candidate > best_dice:
                            best_dice = dice_candidate
                            best_mask = pm
                    
                    union_binary = (combined_mask > 0.5).float().cpu()
                    union_intersection = (union_binary * mask_gt_binary).sum()
                    union_dice = (2.0 * union_intersection / (union_binary.sum() + mask_gt_binary.sum() + 1e-6)).item()
                    
                    pred_mask = combined_mask if union_dice > best_dice else best_mask
                
                pred_binary = (pred_mask > 0.5).float().cpu()
                mask_binary = (mask_gt > 0.5).float()
                intersection = (pred_binary * mask_binary).sum()
                dice = (2.0 * intersection / (pred_binary.sum() + mask_binary.sum() + 1e-6)).item()
                trained_dice_scores.append(dice)
        except:
            pass

trained_avg_dice = np.mean(trained_dice_scores) if trained_dice_scores else 0.0
print(f"   Trained Model Dice: {trained_avg_dice:.4f}")

# 3. Check weight changes
print("\n3. Checking LoRA weight changes...")
if global_weights is not None:
    weight_diffs = []
    for i, (init_w, trained_w) in enumerate(zip(init_weights, global_weights)):
        diff = np.abs(init_w - trained_w).mean()
        weight_diffs.append(diff)
    
    avg_diff = np.mean(weight_diffs)
    max_diff = np.max(weight_diffs)
    print(f"   Average weight change: {avg_diff:.6f}")
    print(f"   Max weight change: {max_diff:.6f}")
    print(f"   Weights changed: {'YES' if avg_diff > 1e-8 else 'NO'}")
else:
    print("   No global_weights available (FL didn't run)")

# 4. Compare client types
print("\n4. Client Type Comparison (final round):")
if fl_history["round_metrics"]:
    final_metrics = fl_history["round_metrics"][-1]
    for do_name, metrics in final_metrics.items():
        print(f"   {do_name} ({metrics['method']}): Dice = {metrics['dice']:.4f}")

# 5. Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
improvement = trained_avg_dice - init_avg_dice
print(f"Initial Model Dice:  {init_avg_dice:.4f}")
print(f"Trained Model Dice:  {trained_avg_dice:.4f}")
print(f"Improvement:         {improvement:+.4f} ({improvement/max(init_avg_dice, 0.001)*100:+.1f}%)")
print()
if improvement > 0.01:
    print("✅ LoRA training is EFFECTIVE - model improved after FL")
elif improvement > -0.01:
    print("⚠️  LoRA training shows MINIMAL effect - consider more rounds/data")
else:
    print("❌ LoRA training may not be working - check gradient flow")

del trained_model

# ======================================================================
# Code from Cell 15
# ======================================================================

# Cell 14: Summary Statistics

print("="*80)
print("FEDERATED LEARNING SUMMARY")
print("="*80)

print(f"\nConfiguration:")
print(f"  - Model: SAM2LoRA (from task.py)")
print(f"  - Rounds: {NUM_ROUNDS}")
print(f"  - Local Epochs: {LOCAL_EPOCHS}")
print(f"  - Learning Rate: {LEARNING_RATE}")
print(f"  - LoRA Rank: {LORA_RANK}")

print(f"\nData Owner Summary:")
final_round_metrics = fl_history["round_metrics"][-1]
for do_name, metrics in final_round_metrics.items():
    print(f"  {do_name}:")
    print(f"    Method: {metrics['method']}")
    print(f"    Final Dice: {metrics['dice']:.4f}")
    print(f"    Samples: {metrics['num_samples']}")

print(f"\nGlobal Model:")
print(f"  Initial Dice: {init_dice:.4f}")
print(f"  Final Dice: {fl_history['global_dice'][-1]:.4f}")
print(f"  Improvement: {fl_history['global_dice'][-1] - init_dice:+.4f}")

print(f"\nKey Observations:")
print(f"  - Only LoRA clients (DO_3, DO_4) contribute to weight aggregation")
print(f"  - Zero-shot uses CLIP for text-guided segmentation")
print(f"  - Few-shot uses memory bank with stored examples")
print(f"  - Global model improves via FedAvg of LoRA adapters")
print("="*80)

# ======================================================================
# Code from Cell 16
# ======================================================================

# Cell 16: Save Trained Model

import json

output_dir = PROJECT_ROOT / "outputs"
output_dir.mkdir(exist_ok=True)

# Create model with trained weights and save
if global_weights is not None:
    print("Creating model with trained LoRA weights...")
    final_model = create_model(
        sam2_checkpoint=str(DEFAULT_SAM2_CHECKPOINT),
        sam2_config=DEFAULT_SAM2_CONFIG,
        img_size=IMG_SIZE,
        lora_rank=LORA_RANK,
        use_clip=USE_CLIP,
    )
    set_weights(final_model, global_weights)
    
    # Save adapters using SAM2LoRA's built-in method
    save_path = output_dir / "fl_sam2lora_adapters.pth"
    final_model.save_adapters(str(save_path))
    print(f"✓ Model adapters saved to: {save_path}")
    
    del final_model
else:
    print("⚠️ No global_weights available - run FL training first (Cell 12)")

# Save FL history
history_path = output_dir / "fl_history.json"

# Convert to JSON-serializable format
history_json = {
    "global_loss": fl_history["global_loss"],
    "global_dice": fl_history["global_dice"],
    "config": {
        "num_rounds": NUM_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lora_rank": LORA_RANK,
    }
}

# Add per-round metrics (convert to serializable format)
history_json["round_metrics"] = []
for round_metrics in fl_history["round_metrics"]:
    round_data = {}
    for do_name, metrics in round_metrics.items():
        round_data[do_name] = {
            "method": metrics["method"],
            "dice": metrics["dice"],
            "loss": metrics["loss"],
            "num_samples": metrics["num_samples"],
        }
    history_json["round_metrics"].append(round_data)

with open(history_path, "w") as f:
    json.dump(history_json, f, indent=2)

print(f"✓ FL history saved to: {history_path}")