# Federated SAM2 Medical Image Segmentation

Federated learning for medical image segmentation using [SAM2](https://github.com/facebookresearch/sam2) with [LoRA](https://arxiv.org/pdf/2106.09685) adapters. Inspired by [SAM 2 Few-Shot/Zero-Shot Segmentation](https://github.com/ParallelLLC/Segmentation).

## Overview

This project enables privacy-preserving medical image segmentation across multiple hospitals using:
- **SAM2 with LoRA**: Segment Anything Model 2 with lightweight Low-Rank Adaptation adapters
- **Federated Learning**: Train across distributed datasets without sharing raw data
- **P2P via Google Drive**: Peer-to-peer communication using Google Drive as communication layer (powered by [syft-flwr](https://github.com/OpenMined/syft-flwr) and [syft-client](https://github.com/OpenMined/syft-client))

## Key Features

- **Privacy-Preserving**: Raw medical images never leave the hospital/data owner
- **Lightweight Communication**: Only LoRA adapters (~2-8 MB) are transferred, not the full model
- **Google Colab Compatible**: Run on free Colab instances with GPU support
- **Small Data**: Effective with as few as 20 annotated samples per site

## Project Structure

```
fl-sam2-lora/
├── README.md
├── scripts/                    # Utility scripts
│   ├── download_dataset.py     # Download Chest CT dataset from Kaggle
│   └── run_local_fl.py         # Standalone FL training script
├── notebooks/                  # Jupyter notebooks
│   ├── local.ipynb             # Local FL simulation (main notebook)
│   ├── do.ipynb                # Data Owner notebook
│   ├── ds.ipynb                # Data Scientist notebook
│   └── sam2.ipynb              # SAM2 exploration
├── fl-sam2-lora/               # Flower FL application
│   └── fl-sam2-lora/
│       ├── task.py             # SAM2LoRA model implementation
│       ├── client_app.py       # Flower client
│       └── server_app.py       # Flower server
├── docs/                       # Documentation
├── libs/                       # External libraries
│   └── sam2/                   # SAM2 library
└── dataset/                    # Dataset directory (not tracked)
```

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repo-url>
cd fl-sam2-lora

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Configure your Kaggle credentials in `~/.kaggle/kaggle.json`, then:

```bash
python scripts/download_dataset.py
```

This downloads the [Chest CT Segmentation](https://www.kaggle.com/datasets/polomarco/chest-ct-segmentation) dataset.

### 3. Run Local FL Simulation

**Option A: Jupyter Notebook (recommended for exploration)**
```bash
jupyter notebook notebooks/local.ipynb
```

**Option B: Standalone Script (for headless execution)**
```bash
python scripts/run_local_fl.py
```

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/download_dataset.py` | Downloads and organizes the Chest CT Segmentation dataset from Kaggle |
| `scripts/run_local_fl.py` | Standalone FL training script with 4 simulated Data Owners (zero-shot, few-shot, 2x LoRA) |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/local.ipynb` | Main notebook for local FL simulation with heterogeneous clients |
| `notebooks/do.ipynb` | Data Owner perspective notebook |
| `notebooks/ds.ipynb` | Data Scientist perspective notebook |
| `notebooks/sam2.ipynb` | SAM2 model exploration and testing |

## References

- [SAM 2 Few-Shot/Zero-Shot Segmentation](https://github.com/ParallelLLC/Segmentation)
- [syft-flwr](https://github.com/OpenMined/syft-flwr)
- [syft-client](https://github.com/OpenMined/syft-client)
- [SAM2 (Segment Anything Model 2)](https://github.com/facebookresearch/segment-anything-2)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Flower Federated Learning](https://flower.ai/)
- [SyftBox](https://syftbox.net/)
