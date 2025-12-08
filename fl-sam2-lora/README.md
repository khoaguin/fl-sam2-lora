# FL SAM2 Segmentation

Federated Learning project for medical image segmentation using SAM2 with LoRA adapters.

## Overview

This Flower-based FL project trains SAM2 LoRA adapters across distributed medical imaging datasets without sharing raw data.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      SAM2LoRALite Model                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input Image [B, 3, H, W]                                         │
│       │                                                           │
│       ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │          Patch Embedding + Position Embedding              │  │
│  │                    (Frozen)                                  │  │
│  └──────────────────────────┬──────────────────────────────────┘  │
│                             │                                     │
│                             ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │        Transformer Encoder (12 layers)                      │  │
│  │        ┌────────────────────────────────────────┐            │  │
│  │        │ Attention with LoRA                    │            │  │
│  │        │   - Q, K, V projections (LoRA)         │ ← Trainable │  │
│  │        │   - Output projection (LoRA)           │            │  │
│  │        └────────────────────────────────────────┘            │  │
│  └──────────────────────────┬──────────────────────────────────┘  │
│                             │                                     │
│                             ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │              Mask Decoder (Upsampling)                      │  │
│  │                    (Frozen)                                  │  │
│  └──────────────────────────┬──────────────────────────────────┘  │
│                             │                                     │
│                             ▼                                     │
│                   Output Mask [B, 1, H, W]                        │
└─────────────────────────────────────────────────────────────────┘

Trainable parameters: ~400K (LoRA adapters only)
Communication per round: ~2-8 MB
```

## Files

- `main.py`: Entry point for SyftBox execution
- `pyproject.toml`: Project configuration and dependencies
- `fl_sam2_lora/`:
  - `task.py`: Model definition, data loading, train/evaluate functions
  - `client_app.py`: Flower client for Data Owners
  - `server_app.py`: Flower server for aggregation

## Usage

### Local Simulation

```bash
# Install dependencies
uv pip install -e .

# Run simulation
flwr run .
```

### With SyftBox P2P

```bash
# Set environment variables
export SYFTBOX_EMAIL="your@email.com"
export SYFTBOX_FOLDER="/path/to/syftbox"

# Run
python main.py
```

## Configuration

Edit `pyproject.toml` to customize:

```toml
[tool.flwr.app.config]
num-server-rounds = 3    # FL rounds
modality = "ct"          # ct, mri, ultrasound, histopathology, xray
target-size = 512        # Image size
local-epochs = 3         # Local epochs per round
learning-rate = 0.0001   # LoRA learning rate
```

## Dataset Requirements

```
dataset/
├── train/
│   ├── images/*.png
│   └── masks/*.png
└── test/
    ├── images/*.png
    └── masks/*.png
```

## Performance

Expected performance with 20 samples per site, 3 sites, 3 rounds:
- Dice Score: 0.65-0.75 (depends on dataset)
- Communication: ~6-24 MB total (2-8 MB × 3 rounds)
- Time: ~10-30 min on Colab GPU
