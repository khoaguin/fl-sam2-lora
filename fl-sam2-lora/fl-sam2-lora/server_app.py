"""
Flower Server App for Federated Medical Image Segmentation with SAM2 LoRA.

This module implements the server-side federated learning logic for
the Data Scientist (aggregator).
"""

import os
from pathlib import Path

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from fl_sam2_segmentation.task import (
    create_model,
    get_weights,
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CONFIG,
)


def weighted_dice_average(metrics):
    """
    Aggregate Dice scores from all clients weighted by number of samples.

    This provides a fair average across clients with different dataset sizes.
    """
    print("\n" + "=" * 80)
    print("AGGREGATING METRICS FROM CLIENTS")
    print(f"   Number of clients: {len(metrics)}")
    print("=" * 80)

    # Aggregate Dice scores
    dice_scores = []
    examples = []

    for num_examples, m in metrics:
        if "dice" in m:
            dice_scores.append(num_examples * m["dice"])
            examples.append(num_examples)
        elif "train_dice" in m:
            dice_scores.append(num_examples * m["train_dice"])
            examples.append(num_examples)

    if examples:
        avg_dice = sum(dice_scores) / sum(examples)
        print(f" AGGREGATION COMPLETE - Average Dice Score: {avg_dice:.4f}\n")
        return {"dice": avg_dice}
    else:
        print(" No metrics to aggregate\n")
        return {}


def server_fn(context: Context) -> ServerAppComponents:
    """
    Server function to configure federated learning strategy.

    Sets up:
    - Initial global model (LoRA adapters)
    - FedAvg strategy for adapter aggregation
    - Model saving for checkpointing
    """
    print("\n" + "#" * 80)
    print("SAM2 FEDERATED LEARNING SERVER STARTED")
    print(f"   Run Config: {context.run_config}")
    print("#" * 80 + "\n")

    # Get config
    img_size = context.run_config.get("target-size", 1024)
    num_rounds = context.run_config.get("num-server-rounds", 3)
    lora_rank = context.run_config.get("lora-rank", 16)
    use_clip = context.run_config.get("use-clip", True)

    # SAM2 checkpoint configuration
    sam2_checkpoint = os.environ.get("SAM2_CHECKPOINT", DEFAULT_SAM2_CHECKPOINT)
    sam2_config = os.environ.get("SAM2_CONFIG", DEFAULT_SAM2_CONFIG)

    print(f"   SAM2 Checkpoint: {sam2_checkpoint}")
    print(f"   SAM2 Config: {sam2_config}")

    # Create initial model and get parameters
    print(" Creating initial SAM2LoRA model...")
    model = create_model(
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
        img_size=img_size,
        lora_rank=lora_rank,
        use_clip=use_clip,
    )
    initial_params = ndarrays_to_parameters(get_weights(model))

    # Setup output directory for model saving
    from syft_flwr.strategy import FedAvgWithModelSaving

    output_dir = os.getenv("OUTPUT_DIR")
    if output_dir is None:
        output_dir = Path.home() / ".syftbox/rds/"
        output_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(output_dir) / "sam2_lora_weights"

    # Get strategy parameters
    min_available_clients = context.run_config.get("min-available-clients", 1)
    min_fit_clients = context.run_config.get("min-fit-clients", 1)
    min_evaluate_clients = context.run_config.get("min-evaluate-clients", 1)
    fraction_fit = context.run_config.get("fraction-fit", 1.0)
    fraction_evaluate = context.run_config.get("fraction-evaluate", 1.0)

    print(" CONFIGURING FEDAVG STRATEGY")
    print(f"   Model save path: {save_path}")
    print(f"   Number of rounds: {num_rounds}")
    print(f"   Min available clients: {min_available_clients}")
    print(f"   Min fit clients: {min_fit_clients}")
    print(f"   Min evaluate clients: {min_evaluate_clients}")
    print(f"   Fraction fit: {fraction_fit}")
    print(f"   Fraction evaluate: {fraction_evaluate}")

    # Create FedAvg strategy with model saving
    strategy = FedAvgWithModelSaving(
        save_path=save_path,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=min_available_clients,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        initial_parameters=initial_params,
        fit_metrics_aggregation_fn=weighted_dice_average,
        evaluate_metrics_aggregation_fn=weighted_dice_average,
        # Custom config to pass to clients
        # Improved defaults for LoRA training: more epochs, slightly lower LR for stability
        on_fit_config_fn=lambda round_num: {
            "local_epochs": context.run_config.get("local-epochs", 5),  # Increased from 3 to 5 for better convergence
            "learning_rate": context.run_config.get("learning-rate", 5e-5),  # Lower LR (5e-5) for more stable LoRA training
            "round": round_num,
        },
    )

    config = ServerConfig(num_rounds=num_rounds)

    print(" SERVER INITIALIZATION COMPLETE\n")

    return ServerAppComponents(config=config, strategy=strategy)


# Create Flower ServerApp
app = ServerApp(server_fn=server_fn)
