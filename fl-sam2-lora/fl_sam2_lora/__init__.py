"""
Federated Learning for Medical Image Segmentation using SAM2 with LoRA.

This package provides a Flower-based federated learning implementation
for privacy-preserving medical image segmentation across hospital sites.

Supports:
- Zero-shot (0 samples): CLIP text prompts only (~35-42% IoU)
- Few-shot (1-5 samples): Memory bank (~60-65% IoU)
- LoRA training (>10 samples): Full fine-tuning (>70% Dice)
"""

from fl_sam2_lora.task import (
    SAM2LoRA,
    create_model,
    get_weights,
    set_weights,
    load_demo_dataset,
    load_syftbox_dataset,
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CONFIG,
    # Dataset classes
    ChestCTDataset,
    AugmentedChestCTDataset,
    # Checkpoint utilities
    validate_and_download_checkpoint,
    SAM2_CHECKPOINT_URLS,
    # Verification utilities
    verify_lora_effectiveness,
)

__all__ = [
    "SAM2LoRA",
    "create_model",
    "get_weights",
    "set_weights",
    "load_demo_dataset",
    "load_syftbox_dataset",
    "DEFAULT_SAM2_CHECKPOINT",
    "DEFAULT_SAM2_CONFIG",
    # Dataset classes
    "ChestCTDataset",
    "AugmentedChestCTDataset",
    # Checkpoint utilities
    "validate_and_download_checkpoint",
    "SAM2_CHECKPOINT_URLS",
    # Verification utilities
    "verify_lora_effectiveness",
]
