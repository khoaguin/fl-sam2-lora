# Federated SAM2 Medical Image Segmentation - Colab Notebooks

Federated learning on medical image segmentation using [SAM2](https://github.com/facebookresearch/sam2) model with [LoRA](https://arxiv.org/pdf/2106.09685) adapters via Google Drive and Colab. Inspired by [SAM 2 Few-Shot/Zero-Shot Segmentation](https://github.com/ParallelLLC/Segmentation) project.

## Overview

This project enables privacy-preserving medical image segmentation across multiple hospitals using:
- **SAM2 with LoRA**: Segment Anything Model 2 with lightweight Low-Rank Adaptation adapters
- **Federated Learning**: Train across distributed datasets without sharing raw data
- **P2P via Google Drive**: Peer-to-peer communication Google Drive as communication layer (powered by [syft-flwr](https://github.com/OpenMined/syft-flwr) and [syft-client](https://github.com/OpenMined/syft-client))

## Key Features

- **Privacy-Preserving**: Raw medical images never leave the hospital/data owner
- **Lightweight Communication**: Only LoRA adapters (~2-8 MB) are transferred, not the full model
- **Google Colab Compatible**: Run on free Colab instances with GPU support
- **Small Data**: Effective with as few as 20 annotated samples per site

## References
- [SAM 2 Few-Shot/Zero-Shot Segmentation](https://github.com/ParallelLLC/Segmentation)
- [syft-flwr](https://github.com/OpenMined/syft-flwr)
- [syft-client](https://github.com/OpenMined/syft-client)
- [SAM2 (Segment Anything Model 2)](https://github.com/facebookresearch/segment-anything-2)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Flower Federated Learning](https://flower.ai/)
- [SyftBox](https://syftbox.net/)
