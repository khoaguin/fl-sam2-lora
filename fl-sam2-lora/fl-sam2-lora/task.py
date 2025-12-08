"""
SAM2 LoRA Task Module for Federated Medical Image Segmentation.

This module provides:
- SAM2LoRALite: Lightweight SAM2 with LoRA adapters for federated learning
- Data loading utilities for medical imaging datasets
- Training and evaluation functions

Designed for Google Colab with limited GPU memory.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
from loguru import logger
from torch.utils.data import DataLoader, Dataset


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_device()
logger.info(f"Using device: {DEVICE}")


# ============================================================================
# SAM2 LoRA Lite Model (Simplified for Colab)
# ============================================================================

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer for efficient fine-tuning.

    This wraps a frozen linear layer and adds trainable low-rank matrices.
    Only the LoRA parameters are updated during training.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen original weight (will be loaded from pretrained model)
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features), requires_grad=False
        )
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)

        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward pass (frozen)
        original_output = F.linear(x, self.weight, self.bias)

        # LoRA forward pass (trainable)
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T

        return original_output + lora_output * self.scaling


class SAM2LoRALite(nn.Module):
    """
    Lightweight SAM2-like model with LoRA adapters for federated learning.

    This is a simplified version suitable for Google Colab with limited memory.
    Uses a Vision Transformer (ViT) encoder with LoRA adapters and a
    lightweight mask decoder.

    Architecture:
    - ViT-Small encoder (frozen backbone)
    - LoRA adapters on attention layers (trainable, ~2-8 MB)
    - Simple mask decoder

    For production use, replace with full SAM2 model from Meta.
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        embed_dim: int = 384,  # ViT-Small
        depth: int = 12,
        num_heads: int = 6,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        num_classes: int = 1,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer encoder with LoRA
        self.encoder_layers = nn.ModuleList([
            TransformerBlockWithLoRA(
                dim=embed_dim,
                num_heads=num_heads,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
            for _ in range(depth)
        ])

        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Lightweight mask decoder
        self.decoder = MaskDecoder(
            embed_dim=embed_dim,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
        )

        # Freeze backbone, keep LoRA trainable
        self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze all non-LoRA parameters."""
        for name, param in self.named_parameters():
            if 'lora' not in name.lower():
                param.requires_grad = False
            else:
                param.requires_grad = True

    def get_adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get only the LoRA adapter weights for federated aggregation."""
        adapter_state = {}
        for name, param in self.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                adapter_state[name] = param.data.cpu().clone()
        return adapter_state

    def load_adapter_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load LoRA adapter weights from aggregated state."""
        for name, param in self.named_parameters():
            if name in state_dict:
                param.data.copy_(state_dict[name].to(param.device))

    def get_trainable_parameters(self):
        """Get list of trainable parameters (LoRA only)."""
        return [p for p in self.parameters() if p.requires_grad]

    def forward(
        self,
        x: torch.Tensor,
        point_prompts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for segmentation.

        Args:
            x: Input image tensor [B, 3, H, W]
            point_prompts: Optional point prompts [B, N, 2] (x, y coordinates)

        Returns:
            Segmentation mask [B, num_classes, H, W]
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        x = x + self.pos_embed

        # Transformer encoder with LoRA
        for layer in self.encoder_layers:
            x = layer(x)

        x = self.encoder_norm(x)

        # Extract patch features (remove CLS token)
        patch_features = x[:, 1:]  # [B, num_patches, embed_dim]

        # Decode to mask
        mask = self.decoder(patch_features, point_prompts)

        return mask


class TransformerBlockWithLoRA(nn.Module):
    """Transformer block with LoRA adapters on attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AttentionWithLoRA(
            dim=dim,
            num_heads=num_heads,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AttentionWithLoRA(nn.Module):
    """Multi-head attention with LoRA adapters on Q, K, V projections."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection with LoRA
        self.qkv = LoRALinear(
            dim, dim * 3,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=dropout,
        )

        # Output projection with LoRA
        self.proj = LoRALinear(
            dim, dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=dropout,
        )

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class MaskDecoder(nn.Module):
    """Simple mask decoder for segmentation."""

    def __init__(
        self,
        embed_dim: int,
        img_size: int,
        patch_size: int,
        num_classes: int = 1,
    ):
        super().__init__()
        self.num_patches_side = img_size // patch_size
        self.embed_dim = embed_dim
        self.img_size = img_size

        # Upsampling path
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(
        self,
        patch_features: torch.Tensor,
        point_prompts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = patch_features.shape[0]

        # Reshape to spatial format
        x = patch_features.transpose(1, 2).reshape(
            B, self.embed_dim, self.num_patches_side, self.num_patches_side
        )

        # Decode to mask
        mask = self.decoder(x)

        # Resize to original size
        mask = F.interpolate(
            mask, size=(self.img_size, self.img_size),
            mode='bilinear', align_corners=False
        )

        return mask


# ============================================================================
# Data Loading Utilities
# ============================================================================

class MedicalSegmentationDataset(Dataset):
    """
    Dataset for medical image segmentation.

    Expects directory structure:
    data_dir/
        images/
            image1.png
            image2.png
            ...
        masks/
            image1.png
            image2.png
            ...
    """

    def __init__(
        self,
        data_dir: Path,
        target_size: int = 512,
        modality: str = "ct",
    ):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.modality = modality

        # Find all images
        images_dir = self.data_dir / "images"
        masks_dir = self.data_dir / "masks"

        self.image_paths = sorted(list(images_dir.glob("*.png")) +
                                   list(images_dir.glob("*.jpg")))
        self.mask_paths = []

        for img_path in self.image_paths:
            mask_path = masks_dir / img_path.name
            if mask_path.exists():
                self.mask_paths.append(mask_path)
            else:
                # Try with different extension
                mask_path = masks_dir / (img_path.stem + ".png")
                self.mask_paths.append(mask_path)

        logger.info(f"Found {len(self.image_paths)} image-mask pairs")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply modality-specific normalization."""
        if self.modality == "ct":
            # CT windowing (soft tissue)
            image = np.clip(image, -150, 250)
            image = (image + 150) / 400
        elif self.modality == "mri":
            # Percentile normalization
            p1, p99 = np.percentile(image, [1, 99])
            image = np.clip(image, p1, p99)
            image = (image - p1) / (p99 - p1 + 1e-8)
        else:
            # Generic normalization
            if image.max() > 1.0:
                image = image / 255.0
        return image

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = image.resize((self.target_size, self.target_size), Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0

        # Load mask
        mask = Image.open(self.mask_paths[idx]).convert("L")
        mask = mask.resize((self.target_size, self.target_size), Image.NEAREST)
        mask = np.array(mask).astype(np.float32) / 255.0

        # Convert to tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # [3, H, W]
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "path": str(self.image_paths[idx]),
        }


def load_syftbox_dataset(
    target_size: int = 512,
    modality: str = "ct",
) -> Tuple[DataLoader, DataLoader]:
    """
    Load medical segmentation dataset from SyftBox.

    Returns train and test dataloaders.
    """
    try:
        import syft_client as sc

        logger.info("[P2P TRANSPORT] Using syft_client to load dataset")

        # Resolve syft paths
        train_path = sc.resolve_path(
            "syft://private/syft_datasets/medical-segmentation/train"
        )
        test_path = sc.resolve_path(
            "syft://private/syft_datasets/medical-segmentation/test"
        )

    except (ImportError, Exception) as e:
        logger.info(f"[SYFTBOX TRANSPORT] Falling back to DATA_DIR ({e})")
        from syft_flwr.utils import get_syftbox_dataset_path

        data_dir = get_syftbox_dataset_path()
        train_path = data_dir / "train"
        test_path = data_dir / "test"

    # Create datasets
    train_dataset = MedicalSegmentationDataset(
        train_path, target_size=target_size, modality=modality
    )
    test_dataset = MedicalSegmentationDataset(
        test_path, target_size=target_size, modality=modality
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # Colab compatibility
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader


def load_demo_dataset(
    num_samples: int = 20,
    target_size: int = 512,
) -> Tuple[DataLoader, DataLoader]:
    """
    Generate synthetic demo dataset for testing.

    Creates random images with circular/elliptical masks.
    """
    logger.info(f"Generating {num_samples} synthetic samples...")

    images = []
    masks = []

    for i in range(num_samples):
        # Generate random image
        image = torch.randn(3, target_size, target_size) * 0.1 + 0.5
        image = image.clamp(0, 1)

        # Generate random mask (circle or ellipse)
        mask = torch.zeros(1, target_size, target_size)

        # Random center and radius
        cx = np.random.randint(target_size // 4, 3 * target_size // 4)
        cy = np.random.randint(target_size // 4, 3 * target_size // 4)
        rx = np.random.randint(30, target_size // 4)
        ry = np.random.randint(30, target_size // 4)

        # Create mask
        y, x = torch.meshgrid(
            torch.arange(target_size), torch.arange(target_size), indexing='ij'
        )
        ellipse = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1
        mask[0] = ellipse.float()

        # Add mask region to image (make it brighter)
        image = image + mask * 0.3
        image = image.clamp(0, 1)

        images.append(image)
        masks.append(mask)

    # Create tensors
    images_tensor = torch.stack(images)
    masks_tensor = torch.stack(masks)

    # Split into train/test
    split = int(0.8 * num_samples)

    train_data = torch.utils.data.TensorDataset(
        images_tensor[:split], masks_tensor[:split]
    )
    test_data = torch.utils.data.TensorDataset(
        images_tensor[split:], masks_tensor[split:]
    )

    train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=2, shuffle=False)

    logger.info(f"Created {split} train and {num_samples - split} test samples")

    return train_loader, test_loader


# ============================================================================
# Training and Evaluation
# ============================================================================

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Dice loss for segmentation."""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def dice_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Dice score."""
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    if union == 0:
        return 1.0
    return (2.0 * intersection / union).item()


def train(
    model: nn.Module,
    train_loader: DataLoader,
    local_epochs: int = 3,
    learning_rate: float = 1e-4,
) -> Dict[str, List[float]]:
    """
    Train the model locally.

    Args:
        model: SAM2LoRALite model
        train_loader: Training dataloader
        local_epochs: Number of local epochs
        learning_rate: Learning rate

    Returns:
        Training history with loss and dice scores
    """
    model.to(DEVICE)
    model.train()

    # Only optimize LoRA parameters
    optimizer = optim.AdamW(
        model.get_trainable_parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    history = {"train_loss": [], "train_dice": []}

    for epoch in range(local_epochs):
        epoch_loss = 0.0
        epoch_dice = 0.0
        num_batches = 0

        for batch in train_loader:
            # Handle both dict and tuple formats
            if isinstance(batch, dict):
                images = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)
            else:
                images, masks = batch
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass
            pred_masks = model(images)

            # Compute loss
            loss = dice_loss(pred_masks, masks)
            loss += F.binary_cross_entropy_with_logits(pred_masks, masks)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_dice += dice_score(pred_masks, masks)
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches
        history["train_loss"].append(avg_loss)
        history["train_dice"].append(avg_dice)

        logger.info(f"Epoch {epoch + 1}/{local_epochs}: Loss={avg_loss:.4f}, Dice={avg_dice:.4f}")

    return history


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
) -> Tuple[float, float]:
    """
    Evaluate the model.

    Args:
        model: SAM2LoRALite model
        test_loader: Test dataloader

    Returns:
        Tuple of (average loss, average dice score)
    """
    model.to(DEVICE)
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                images = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)
            else:
                images, masks = batch
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

            pred_masks = model(images)

            loss = dice_loss(pred_masks, masks)
            loss += F.binary_cross_entropy_with_logits(pred_masks, masks)

            total_loss += loss.item()
            total_dice += dice_score(pred_masks, masks)
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_dice = total_dice / num_batches if num_batches > 0 else 0.0

    return avg_loss, avg_dice


def get_weights(model: nn.Module) -> List[np.ndarray]:
    """Get model weights as numpy arrays (LoRA adapters only)."""
    adapter_dict = model.get_adapter_state_dict()
    return [v.cpu().numpy() for v in adapter_dict.values()]


def set_weights(model: nn.Module, parameters: List[np.ndarray]):
    """Set model weights from numpy arrays."""
    adapter_dict = model.get_adapter_state_dict()
    keys = list(adapter_dict.keys())
    new_state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_adapter_state_dict(new_state)


# ============================================================================
# Model Factory
# ============================================================================

def create_model(
    img_size: int = 512,
    lora_rank: int = 8,
) -> SAM2LoRALite:
    """
    Create a new SAM2LoRALite model.

    Args:
        img_size: Input image size
        lora_rank: LoRA rank (higher = more capacity, larger size)

    Returns:
        Initialized model
    """
    model = SAM2LoRALite(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,  # ViT-Small
        depth=12,
        num_heads=6,
        lora_rank=lora_rank,
        lora_alpha=16.0,
        num_classes=1,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter_size_mb = (trainable_params * 4) / (1024 ** 2)

    logger.info(f"Model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable (LoRA) parameters: {trainable_params:,}")
    logger.info(f"  LoRA adapter size: {adapter_size_mb:.2f} MB")

    return model
