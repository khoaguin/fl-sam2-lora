"""
Flower Client App for Federated Medical Image Segmentation with SAM2 LoRA.

This module implements the client-side federated learning logic for
Data Owners (hospitals/medical institutions).
"""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from loguru import logger

from fl_sam2_segmentation.task import (
    create_model,
    evaluate,
    get_weights,
    set_weights,
    train,
    load_demo_dataset,
)


class SAM2LoRAClient(NumPyClient):
    """
    Flower client wrapping SAM2LoRALite for federated learning.

    This client:
    - Loads local medical imaging data
    - Trains LoRA adapters locally (data never leaves the site)
    - Sends only LoRA adapter weights (~2-8 MB) for aggregation
    - Evaluates model on local validation data
    """

    def __init__(self, model, train_loader, test_loader):
        print("\n" + "=" * 80)
        print("SAM2 LORA CLIENT INITIALIZED")
        print(f"   Training batches: {len(train_loader)} | Test batches: {len(test_loader)}")
        print("=" * 80 + "\n")

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self, config):
        """Return LoRA adapter weights as numpy arrays."""
        return get_weights(self.model)

    def fit(self, parameters, config):
        """
        Train LoRA adapters locally on private medical data.

        Data never leaves this site - only adapter weights are returned.
        """
        print("\n" + ">" * 80)
        print("TRAINING ROUND STARTED")
        print(f"   Batches: {len(self.train_loader)}")
        print(">" * 80)

        # Load global adapter weights
        set_weights(self.model, parameters)

        # Get training config
        local_epochs = config.get("local_epochs", 3)
        learning_rate = config.get("learning_rate", 1e-4)

        # Train locally
        history = train(
            self.model,
            self.train_loader,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
        )

        print(f" TRAINING COMPLETE")
        print(f"   Final Loss: {history['train_loss'][-1]:.4f}")
        print(f"   Final Dice: {history['train_dice'][-1]:.4f}\n")

        # Return updated adapter weights
        return (
            get_weights(self.model),
            len(self.train_loader.dataset),
            {"train_loss": history["train_loss"][-1], "train_dice": history["train_dice"][-1]},
        )

    def evaluate(self, parameters, config):
        """
        Evaluate model on local test data.

        Returns metrics without exposing raw data.
        """
        print("\n" + "*" * 80)
        print("EVALUATION ROUND STARTED")
        print(f"   Batches: {len(self.test_loader)}")
        print("*" * 80)

        # Load global adapter weights
        set_weights(self.model, parameters)

        # Evaluate locally
        loss, dice = evaluate(self.model, self.test_loader)

        print(" EVALUATION RESULTS:")
        print(f"   Loss: {loss:.4f} | Dice Score: {dice:.4f}")
        print(" EVALUATION COMPLETE\n")

        return loss, len(self.test_loader.dataset), {"dice": dice}


def client_fn(context: Context):
    """
    Factory function to create SAM2LoRA client.

    Called by Flower framework to instantiate a client.
    """
    print("\n" + "#" * 80)
    print("SAM2 CLIENT FUNCTION STARTED")
    print(f"   Node Config: {context.node_config}")
    print("#" * 80 + "\n")

    from syft_flwr.utils import run_syft_flwr

    # Get config
    img_size = context.run_config.get("target-size", 512)
    modality = context.run_config.get("modality", "ct")

    # Create model
    model = create_model(img_size=img_size, lora_rank=8)

    # Load data
    if not run_syft_flwr():
        # Local simulation mode - use demo data
        print(" Loading demo dataset locally...")
        logger.info("Running flwr locally with demo data")
        train_loader, test_loader = load_demo_dataset(
            num_samples=20,
            target_size=img_size,
        )
    else:
        # SyftBox mode - load real data
        print(" Loading SyftBox dataset...")
        logger.info("Running with syft_flwr")
        from fl_sam2_segmentation.task import load_syftbox_dataset
        train_loader, test_loader = load_syftbox_dataset(
            target_size=img_size,
            modality=modality,
        )

    return SAM2LoRAClient(model, train_loader, test_loader).to_client()


# Create Flower ClientApp
app = ClientApp(client_fn=client_fn)
