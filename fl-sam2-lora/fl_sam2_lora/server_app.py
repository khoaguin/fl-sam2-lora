"""
Flower Server App for Federated Medical Image Segmentation with SAM2 LoRA.

This module implements the server-side federated learning logic for
the Data Scientist (aggregator).

Supports heterogeneous clients:
- Zero-shot clients (no data): Participate but don't contribute weights
- Few-shot clients (1-5 samples): Participate but don't contribute weights
- LoRA clients (>10 samples): Full training, contribute weights to FedAvg
"""

import os
from pathlib import Path

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from fl_sam2_lora.task import (
    create_model,
    get_weights,
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CONFIG,
)


def weighted_dice_average(metrics):
    """
    Aggregate Dice scores from all clients weighted by number of samples.

    Handles heterogeneous clients:
    - LoRA clients: Weighted by num_samples
    - Zero-shot/Few-shot: Included with weight 1 (for fair comparison)
    """
    print("\n" + "=" * 80)
    print("AGGREGATING METRICS FROM CLIENTS")
    print(f"   Number of clients: {len(metrics)}")
    print("=" * 80)

    # Separate clients by type
    lora_metrics = []
    other_metrics = []

    for num_examples, m in metrics:
        method = m.get("method", "lora")
        dice = m.get("dice", m.get("train_dice", 0.0))

        if method == "lora":
            lora_metrics.append((num_examples, dice))
            print(f"   LoRA client: {num_examples} samples, Dice={dice:.4f}")
        else:
            other_metrics.append((1, dice))  # Weight 1 for non-training clients
            print(f"   {method.upper()} client: Dice={dice:.4f} (no weight contribution)")

    # Compute weighted average
    all_metrics = lora_metrics + other_metrics
    if all_metrics:
        total_weight = sum(w for w, _ in all_metrics)
        weighted_sum = sum(w * d for w, d in all_metrics)
        avg_dice = weighted_sum / total_weight if total_weight > 0 else 0.0
        print(f"\n AGGREGATION COMPLETE")
        print(f"   LoRA clients: {len(lora_metrics)}")
        print(f"   Other clients: {len(other_metrics)}")
        print(f"   Average Dice Score: {avg_dice:.4f}\n")
        return {"dice": avg_dice, "lora_clients": len(lora_metrics), "other_clients": len(other_metrics)}
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

    output_dir = Path(os.getenv("OUTPUT_DIR") or "./models/finetuned")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "sam2_lora_weights"

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

    # Get FedProx configuration
    use_fedprox = context.run_config.get("use-fedprox", True)
    fedprox_mu = context.run_config.get("fedprox-mu", 1e-3)

    print(f"   FedProx: {'Enabled' if use_fedprox else 'Disabled'} (μ={fedprox_mu})")

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
        on_fit_config_fn=lambda round_num: {
            "local_epochs": context.run_config.get("local-epochs", 5),
            "learning_rate": context.run_config.get("learning-rate", 5e-5),
            "round": round_num,
            # FedProx regularization for stable FL training
            "use_fedprox": use_fedprox,
            "fedprox_mu": fedprox_mu,
        },
    )

    config = ServerConfig(num_rounds=num_rounds)

    print(" SERVER INITIALIZATION COMPLETE\n")

    return ServerAppComponents(config=config, strategy=strategy)


# Create Flower ServerApp
app = ServerApp(server_fn=server_fn)
