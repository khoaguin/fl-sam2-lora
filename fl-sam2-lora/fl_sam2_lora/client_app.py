"""
Flower Client App for Federated Medical Image Segmentation with SAM2 LoRA.

This module implements adaptive client-side federated learning logic for
Data Owners (hospitals/medical institutions) with heterogeneous data availability.

Supports:
- Zero-shot (0 samples): CLIP text prompts only
- Few-shot (1-5 samples): Memory bank
- LoRA training (>10 samples): Full fine-tuning
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from loguru import logger

from fl_sam2_lora.task import (
    SAM2LoRA,
    create_model,
    get_weights,
    set_weights,
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CONFIG,
)


class AdaptiveSAM2Client(NumPyClient):
    """
    Adaptive Flower client that selects training mode based on data availability.

    This client:
    - Detects available training data and selects appropriate method
    - Zero-shot (0 samples): Uses CLIP text prompts, returns empty weights
    - Few-shot (1-5 samples): Uses memory bank, returns empty weights
    - LoRA (>10 samples): Full training, returns adapter weights

    Only LoRA clients contribute weights to FedAvg aggregation.
    """

    def __init__(
        self,
        model: SAM2LoRA,
        train_loader: Optional[Any],
        test_loader: Optional[Any],
        modality: str = "ct",
        class_name: str = "target",
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.modality = modality
        self.class_name = class_name

        # Determine client type based on data availability
        if train_loader is None:
            self.num_samples = 0
        else:
            self.num_samples = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else len(train_loader)

        # Determine training method
        if self.num_samples == 0:
            self.client_type = "zero_shot"
        elif self.num_samples <= 5:
            self.client_type = "few_shot"
        else:
            self.client_type = "lora"

        print("\n" + "=" * 80)
        print("ADAPTIVE SAM2 LORA CLIENT INITIALIZED")
        print(f"   Training samples: {self.num_samples}")
        print(f"   Client type: {self.client_type.upper()}")
        print(f"   Modality: {self.modality}")
        if self.client_type == "lora":
            print(f"   -> Will contribute weights to FedAvg")
        else:
            print(f"   -> Will NOT contribute weights (uses {self.client_type} inference)")
        print("=" * 80 + "\n")

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return LoRA adapter weights as numpy arrays."""
        if self.client_type != "lora":
            # Non-LoRA clients return empty weights
            return []
        return get_weights(self.model)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Train using adaptive method selection based on data availability.

        Enhanced features (for LoRA mode):
        - FedProx regularization for stable FL training
        - Background point prompts for better boundary learning
        - Early stopping to prevent overfitting

        Returns:
            - weights: LoRA adapter weights (empty list for non-LoRA)
            - num_samples: Number of training samples
            - metrics: {method, dice, loss, num_samples}
        """
        print("\n" + ">" * 80)
        print(f"TRAINING ROUND STARTED - {self.client_type.upper()} MODE")
        print(f"   Samples: {self.num_samples}")
        print(">" * 80)

        # Get training config with enhanced defaults
        local_epochs = config.get("local_epochs", 5)
        learning_rate = config.get("learning_rate", 1e-4)
        round_num = config.get("round", 1)
        use_fedprox = config.get("use_fedprox", True)
        fedprox_mu = config.get("fedprox_mu", 1e-3)

        # Store global weights for FedProx (before loading into model)
        global_weights_for_fedprox = None

        # Load global weights only for LoRA clients
        if self.client_type == "lora" and len(parameters) > 0:
            try:
                # Validate rank compatibility before loading
                model_lora_rank = getattr(self.model, 'lora_rank', None)
                if model_lora_rank:
                    print(f"   Model LoRA rank: {model_lora_rank}")

                # Store for FedProx before modifying model
                global_weights_for_fedprox = parameters

                set_weights(self.model, parameters)
                print(f"   Loaded global weights from round {round_num - 1}")
            except RuntimeError as e:
                if "size" in str(e).lower() or "dimension" in str(e).lower():
                    logger.error(f"Rank mismatch when loading global weights: {e}")
                    logger.error("Model and global weights have incompatible LoRA ranks")
                    # Continue without loading weights (first round behavior)
                    global_weights_for_fedprox = None
                else:
                    logger.warning(f"Could not load global weights: {e}")
                    global_weights_for_fedprox = None

        # Use adaptive_fit with enhanced features
        result = self.model.adaptive_fit(
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            modality=self.modality,
            class_name=self.class_name,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            few_shot_threshold=5,
            lora_threshold=10,  # 0→zero-shot, 1-5→few-shot, >10→LoRA
            # Enhanced training options
            use_fedprox=use_fedprox,
            fedprox_mu=fedprox_mu,
            global_weights=global_weights_for_fedprox,
            use_background_prompts=True,
            early_stopping_patience=3,
        )

        method = result['method']
        metrics = result['metrics']
        weights = result['weights']

        print(f"\n TRAINING COMPLETE - {method.upper()}")
        print(f"   Dice: {metrics['dice']:.4f}")
        print(f"   Loss: {metrics['loss']:.4f}")

        # Prepare return values
        if method == "lora" and weights is not None:
            return_weights = weights
            print(f"   -> Returning {len(return_weights)} weight arrays for aggregation")
        else:
            return_weights = []
            print(f"   -> No weights to aggregate ({method} mode)")

        return (
            return_weights,
            self.num_samples,
            {
                "method": method,
                "dice": metrics['dice'],
                "loss": metrics['loss'],
                "num_samples": self.num_samples,
            },
        )

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, float]]:
        """
        Evaluate model on local test data.

        Returns metrics without exposing raw data.
        """
        print("\n" + "*" * 80)
        print(f"EVALUATION ROUND - {self.client_type.upper()} MODE")
        print("*" * 80)

        # Load global weights for LoRA clients
        if self.client_type == "lora" and len(parameters) > 0:
            try:
                set_weights(self.model, parameters)
            except Exception as e:
                logger.warning(f"Could not load weights for eval: {e}")

        # Evaluate based on client type
        if self.test_loader is None:
            print("   No test data available")
            return 1.0, 0, {"dice": 0.0, "method": self.client_type}

        self.model.eval()
        dice_scores = []
        test_samples = 0

        import torch
        with torch.no_grad():
            for batch in self.test_loader:
                if isinstance(batch, dict):
                    image = batch["image"][0]
                    mask_gt = batch["mask"][0, 0]
                else:
                    image, mask = batch
                    image = image[0]
                    mask_gt = mask[0, 0]

                try:
                    # Use appropriate segmentation method
                    if self.client_type == "zero_shot":
                        predictions = self.model.zero_shot_segment(
                            image=image,
                            modality=self.modality,
                            class_names=[self.class_name, "tissue"],
                            similarity_threshold=0.1,
                        )
                        if predictions:
                            pred_mask = list(predictions.values())[0]
                        else:
                            pred_mask = None
                    elif self.client_type == "few_shot":
                        predictions = self.model.few_shot_segment(
                            image=image,
                            modality=self.modality,
                            class_names=[self.class_name],
                            top_k=3,
                        )
                        if predictions:
                            pred_mask = list(predictions.values())[0]
                        else:
                            pred_mask = None
                    else:
                        # LoRA - use forward_sam2_differentiable with proper point prompts
                        # Extract point prompts from GT mask (same as training)
                        import torch
                        device = next(self.model.parameters()).device
                        img_size = self.model.img_size
                        H_orig, W_orig = mask_gt.shape[-2:]

                        # Get foreground coordinates from GT mask
                        fg_coords = torch.nonzero(mask_gt > 0.5, as_tuple=False).float()

                        if len(fg_coords) > 0:
                            # Use centroid as primary prompt (most reliable)
                            centroid = fg_coords.mean(dim=0)
                            x_cent = centroid[1].item() * img_size / W_orig
                            y_cent = centroid[0].item() * img_size / H_orig

                            point_coords = torch.tensor([[[x_cent, y_cent]]],
                                                        dtype=torch.float32, device=device)
                            point_labels = torch.ones(1, 1, dtype=torch.int32, device=device)
                        else:
                            # Fallback to center if no foreground
                            point_coords = torch.tensor([[[img_size // 2, img_size // 2]]],
                                                        dtype=torch.float32, device=device)
                            point_labels = torch.ones(1, 1, dtype=torch.int32, device=device)

                        pred_mask = self.model.forward_sam2_differentiable(
                            image, point_coords=point_coords, point_labels=point_labels
                        )
                        pred_mask = pred_mask.squeeze()  # Remove batch and channel dims

                    if pred_mask is not None:
                        pred_binary = (pred_mask > 0.5).float().cpu()
                        mask_binary = (mask_gt > 0.5).float().cpu()

                        intersection = (pred_binary * mask_binary).sum()
                        dice = (2.0 * intersection / (pred_binary.sum() + mask_binary.sum() + 1e-6)).item()
                        dice_scores.append(dice)
                    else:
                        dice_scores.append(0.0)

                    test_samples += 1

                except Exception as e:
                    logger.warning(f"Evaluation error: {e}")
                    dice_scores.append(0.0)
                    test_samples += 1

        avg_dice = np.mean(dice_scores) if dice_scores else 0.0
        loss = 1.0 - avg_dice

        print(f" EVALUATION RESULTS:")
        print(f"   Dice: {avg_dice:.4f} | Loss: {loss:.4f}")
        print(f"   Samples evaluated: {test_samples}")
        print(" EVALUATION COMPLETE\n")

        return loss, test_samples, {"dice": avg_dice, "method": self.client_type}


def client_fn(context: Context):
    """
    Factory function to create adaptive SAM2LoRA client.

    Called by Flower framework to instantiate a client.
    Automatically detects data availability and selects training mode.
    """
    print("\n" + "#" * 80)
    print("ADAPTIVE SAM2 CLIENT FUNCTION STARTED")
    print(f"   Node Config: {context.node_config}")
    print("#" * 80 + "\n")

    from syft_flwr.utils import run_syft_flwr

    # Get config from run_config or environment
    img_size = context.run_config.get("target-size", 1024)
    modality = context.run_config.get("modality", "ct")
    lora_rank = context.run_config.get("lora-rank", 16)
    use_clip = context.run_config.get("use-clip", True)

    # SAM2 checkpoint configuration
    sam2_checkpoint = os.environ.get("SAM2_CHECKPOINT", DEFAULT_SAM2_CHECKPOINT)
    sam2_config = os.environ.get("SAM2_CONFIG", DEFAULT_SAM2_CONFIG)

    print(f"   SAM2 Checkpoint: {sam2_checkpoint}")
    print(f"   SAM2 Config: {sam2_config}")
    print(f"   Image Size: {img_size}")
    print(f"   LoRA Rank: {lora_rank}")

    # Create model with full SAM2 + LoRA
    model = create_model(
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
        img_size=img_size,
        lora_rank=lora_rank,
        use_clip=use_clip,
    )

    # Load data
    train_loader = None
    test_loader = None

    try:
        # SyftBox mode - load real data
        print(" Loading SyftBox dataset...")
        from fl_sam2_lora.task import load_syftbox_dataset
        train_loader, test_loader = load_syftbox_dataset(
            target_size=img_size,
            modality=modality,
        )
    except Exception as e:
        logger.warning(f"Could not load SyftBox dataset: {e}")
        logger.info("Falling back to zero-shot mode (no training data)")
        train_loader = None
        test_loader = None

    return AdaptiveSAM2Client(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        modality=modality,
        class_name="target",
    ).to_client()


# Create Flower ClientApp
app = ClientApp(client_fn=client_fn)
