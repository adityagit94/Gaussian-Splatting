# Environment (reference machine)

This repo was built and verified on:

- OS: WSL2 Ubuntu 22.04
- GPU: NVIDIA RTX 5090
- Driver: 576.88
- CUDA (driver-reported): 12.9
- PyTorch: 2.8.0+cu128 (CUDA 12.8 wheels)
- COLMAP: 3.14.0.dev0 (Commit fe411191 on 2026-02-12) with CUDA

Reference commands used:
- PyTorch install:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

- COLMAP version check:
  colmap -h | head -n 3
