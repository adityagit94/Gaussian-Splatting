# 🎯 3D Gaussian Splatting – Real-Time Scene Reconstruction

<p align="center">
  <img src="https://img.shields.io/badge/Platform-WSL2-blue" />
  <img src="https://img.shields.io/badge/CUDA-12.8-green" />
  <img src="https://img.shields.io/badge/PyTorch-2.8.0-orange" />
  <img src="https://img.shields.io/badge/COLMAP-CUDA--Enabled-purple" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

<p align="center">
  <strong>Transform photos into interactive 3D scenes</strong> — High-fidelity, real-time rendering at <strong>≥30fps @ 1080p</strong>
</p>

<p align="center">
  A production-ready implementation of 3D Gaussian Splatting with CUDA-optimized COLMAP and modern web UI for Windows developers.
</p>

---

## ⚡ Why Gaussian Splatting?

| Feature | Gaussian Splatting | Neural Radiance Fields (NeRF) | Traditional MVS |
|---------|-------------------|-------------------------------|-----------------|
| **Training Time** | ~30 min | 12-24 hours | Hours |
| **Render Speed** | 30+ fps @ 1080p | 1-2 fps | Varies |
| **VRAM Required** | ~8-24 GB | ~24-48 GB | Low |
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Real-Time Viewer** | ✅ Yes | ❌ No | ✅ Sometimes |

**Bottom line**: Get NeRF-quality rendering at 60-100x faster speeds with significantly lower memory footprint.

---

## ✨ Key Capabilities

- **🔬 ML-powered reconstruction** – Uses sparse points from COLMAP to train anisotropic 3D Gaussians
- **⚡ Real-time rendering** – OpenGL viewer with VR support (OpenXR)
- **🎛️ Web-based UI** – FastAPI dashboard for job management and visualization
- **🔧 Fully automated** – One-command setup on WSL2 + automatic COLMAP CUDA compilation
- **📊 Advanced optimizations** – Depth regularization, anti-aliasing, exposure compensation
- **🎓 Research-backed** – Based on SIGGRAPH 2023 paper from Graphdeco-Inria

---

## 🎯 Use Cases

| Use Case | Why GS? | Output |
|----------|---------|--------|
| **AR Applications** | Real-time rendering = instant preview | Mobile-optimized .ply models |
| **Digital Heritage** | Fast training for museum/artifact scanning | Interactive 3D archives |
| **Product Visualization** | High quality + fast iteration | E-commerce 3D views |
| **VR/Metaverse** | OpenXR support + real-time perf | VR-ready scenes |
| **Robotics/Perception** | Fast reconstruction from video | Scene understanding models |
| **Game Development** | Pre-baked high-quality assets | Environment meshes & textures |

---

## 📈 Performance Benchmarks

*Reference: RTX 5090

- **Training**: 20-35 min per scene (60K iterations)
- **Rendering**: 60+ fps @ 1080p (complex scenes)
- **Model compression**: ~200-500 MB per scene
- **Quality**: ~25 PSNR, competitive with NeRF on Mip-NeRF360 datasets

---

# 🚀 Quick Start (5 Minutes)

```bash
# 1️⃣ Clone
 git clone --recursive https://github.com/adityagit94/Gaussian-Splatting.git
 cd Gaussian-Splatting

# 2️⃣ Install system dependencies
 chmod +x scripts/install_system_deps_wsl.sh
 ./scripts/install_system_deps_wsl.sh

# 3️⃣ Create environment
 conda create -n gs python=3.10 -y
 conda activate gs

# 4️⃣ Install PyTorch (CUDA 12.8 wheels)
 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 5️⃣ Install Python deps
 pip install -r requirements.txt

# 6️⃣ Start Local UI
 cd gs_platform/app
 uvicorn server:app --host 0.0.0.0 --port 7860
```

Open: http://127.0.0.1:7860

---

# 📸 Platform Preview

<img width="7620" height="4448" alt="127 0 0 1_7860_(High res)" src="https://github.com/user-attachments/assets/17738385-1956-4c4a-b43f-855f89ee39fe" />


---

# 📁 Repository Structure

```
Gaussian-Splatting/
│
├── gaussian-splatting/              # 🧠 Core training optimizer (Graphdeco-Inria)
│   ├── train.py                     # Main training loop
│   ├── render.py                    # Inference & rendering
│   ├── scene/                       # Camera & Gaussian scene management
│   └── submodules/                  # CUDA kernels (rasterization, KNN, SSIM)
│
├── gs_platform/                     # 🌐 Web UI (FastAPI + Uvicorn)
│   └── app/server.py                # Dashboard & API
│
├── scripts/                         # 🛠️ Automation & setup
│   ├── install_system_deps_wsl.sh   # WSL dependencies
│   └── build_colmap_cuda.sh         # COLMAP CUDA compilation
│
├── docs/                            # 📖 Documentation & screenshots
├── requirements.txt
├── environment.md
└── README.md
```

---

## 🔄 Typical Workflow

```
[Photo Collection]
        ↓
    [COLMAP SfM] ←── CUDA-accelerated camera calibration
        ↓
[Sparse Point Cloud]
        ↓
 [GS Training] ←── PyTorch + CUDA kernels (30 min)
        ↓
[Trained Model] ←── ~300 MB .ply file
        ↓
  [Real-time] ←── OpenGL viewer or OpenXR VR
  [Rendering]
```

---

# 🧱 One-Command Setup (Recommended)

You can optionally create a full automated setup script:

```bash
chmod +x scripts/full_setup.sh
./scripts/full_setup.sh
```

Example content of `full_setup.sh`:

```bash
#!/bin/bash
set -e

./scripts/install_system_deps_wsl.sh
conda create -n gs python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

---

# 🔧 System Requirements

### Hardware
| Tier | GPU | VRAM | Use Case |
|------|-----|------|----------|
| **Minimum** | RTX 3060 | 12 GB | Small indoor scenes (~30K points) |
| **Recommended** | RTX 4090 / RTX 6000 | 24 GB | Production work (complex scenes) |
| **High-End** | RTX 5090 / H100 | 48+ GB | Large-scale datasets, batch training |

### Software
- **OS**: Windows 11 with WSL2 (Ubuntu 22.04)
- **CUDA**: 12.8 compatible NVIDIA driver
- **Python**: 3.10 (3.13+ not supported due to CUDA extension build issues)
- **GPU Tools**: `nvidia-smi` must show your GPU with CUDA support

Verify GPU:
```bash
nvidia-smi
```

---

# 🏗 Build COLMAP (CUDA)

```bash
chmod +x scripts/build_colmap_cuda.sh
./scripts/build_colmap_cuda.sh
```

Verify:

```bash
colmap -h | grep CUDA
```

---

# 🧠 Install Gaussian Splatting

```bash
git submodule add https://github.com/graphdeco-inria/gaussian-splatting.git gaussian-splatting
git submodule update --init --recursive

pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization
pip install -e gaussian-splatting/submodules/simple-knn
pip install -e gaussian-splatting/submodules/fused-ssim
```

---

# 🎯 Example Training Command

```bash
SCENE="/mnt/c/gs_data/datasets/bullet"
UNDIST="$SCENE/undistorted_0"
OUT="/mnt/c/gs_data/outputs/bullet_run1"

python gaussian-splatting/train.py -s "$UNDIST" -m "$OUT" \
  --iterations 60000 \
  --resolution 2 \
  --densify_from_iter 800 \
  --densify_until_iter 4000 \
  --densification_interval 250 \
  --densify_grad_threshold 0.0015 \
  --opacity_reset_interval 800 \
  --percent_dense 0.0015 \
  --lambda_dssim 0.3 \
  --random_background
```

### Parameter Guide
| Parameter | Default | Tuning Tips |
|-----------|---------|------------|
| `--iterations` | 60000 | Increase for complex scenes, decrease for quick tests |
| `--resolution` | 1 | Use 2 for large images, 1 for 512×512 input |
| `--densify_grad_threshold` | 0.0015 | Lower = more splats (better quality, slower) |
| `--percent_dense` | 0.0015 | Controls densification rate (critical for floater reduction) |
| `--random_background` | False | ✅ Enable for object-centric scenes |
| `--lambda_dssim` | 0.2 | Increase (0.3-0.5) for better perceptual quality |

---

## 🧹 Recommended Settings (Object-Only Scenes)

To reduce floaters and improve clarity:

```bash
python gaussian-splatting/train.py -s "$UNDIST" -m "$OUT" \
  --iterations 50000 \
  --resolution 2 \
  --densify_until_iter 3500 \
  --percent_dense 0.001 \
  --lambda_dssim 0.3 \
  --random_background \
  --sh_degree 3
```

**Why these settings?**
- Lower `--percent_dense` → Fewer floating artifacts
- Shorter `--densify_until_iter` → Stop growing gaussians early
- `--random_background` → Prevent background collapse
- Higher `--lambda_dssim` → Better structural similarity

---

# � Local Platform UI

```bash
cd gs_platform/app
uvicorn server:app --host 0.0.0.0 --port 7860
```

Then open: **http://127.0.0.1:7860**

**Features**:
- 📊 Real-time training monitoring with loss curves
- 🎥 Render preview & batch export
- 📁 Job management & dataset upload
- 📈 Performance metrics dashboard

---

# 🛠 Troubleshooting

## ❌ CUDA Out of Memory (OOM)

**Symptom**: `RuntimeError: CUDA out of memory`

**Solution 1** – Enable memory allocation optimization:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python gaussian-splatting/train.py ...
```

**Solution 2** – Reduce resolution:
```bash
python gaussian-splatting/train.py ... --resolution 2
```

**Solution 3** – Lower batch size:
```bash
python gaussian-splatting/train.py ... --batch_size 1
```

---

## ❌ CUDA Not Detected

**Check PyTorch CUDA**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Re-install PyTorch with correct CUDA version**:
```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## ❌ COLMAP Build Fails

**Ensure CUDA toolkit is installed**:
```bash
nvcc --version
```

**Force re-build**:
```bash
rm -rf ~/.local/lib/python*/site-packages/colmap*
chmod +x scripts/build_colmap_cuda.sh
./scripts/build_colmap_cuda.sh
```

---

## ❌ Import Errors (submodule compilation)

**Rebuild CUDA extensions**:
```bash
cd gaussian-splatting/submodules/diff-gaussian-rasterization
pip install --force-reinstall -e .
```

Repeat for `simple-knn` and `fused-ssim`.

---

## ❌ Training produces "floater" artifacts

**Increase regularization** (balance quality vs. clarity):
```bash
python gaussian-splatting/train.py ... \
  --percent_dense 0.001 \
  --densify_until_iter 3500 \
  --random_background
```

---

# 📰 Recent Updates

**🆕 October 2024**
- ⚡ Training speed acceleration (2-3x speedup)
- 🎨 Anti-aliasing support
- 🌡️ Depth regularization integration
- 📷 Exposure compensation for varying lighting

**🆕 Spring 2024**
- 🥽 OpenXR support for VR viewing
- 🎯 Improved top-view navigation in SIBR viewer

**📌 Core Features (Stable)**
- Real-time rendering ≥30fps @ 1080p
- Anisotropic covariance optimization
- Density-aware point cloud refinement

---

# 📚 Research & Citation

This implementation is based on:

> **3D Gaussian Splatting for Real-Time Radiance Field Rendering**  
> Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis  
> *ACM Transactions on Graphics* (SIGGRAPH 2023)

**Official resources**:
- 📖 [Full Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf)
- 🔗 [Graphdeco-Inria Repository](https://github.com/graphdeco-inria/gaussian-splatting)
- 🌐 [Project Page](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- 🎥 [Video](https://youtu.be/T_kXY43VZnk)

**Cite this work**:
```bibtex
@Article{kerbl3Dgaussians,
    author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
    title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
    journal      = {ACM Transactions on Graphics},
    number       = {4},
    volume       = {42},
    month        = {July},
    year         = {2023},
    url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

---

# 📌 Environment Reference

See [environment.md](environment.md) for complete setup details.

**Machine specs in my Case**:
- **Driver**: 576.88
- **CUDA** (driver-reported): 12.9
- **PyTorch**: 2.8.0+cu128 (CUDA 12.8 wheels)
- **COLMAP**: 3.14.0.dev0 with CUDA (built Feb 12, 2026)
- **GPU**: NVIDIA RTX 5090 (but RTX 4090, RTX 6000, A6000 also supported)

---

# 📜 License & Attribution

**3D Gaussian Splatting** by Graphdeco-Inria.

This repository provides a **reproducible WSL2 setup + modern web UI wrapper** around the official implementation with:
- ✅ Automated COLMAP CUDA compilation
- ✅ FastAPI dashboard for job management
- ✅ Pre-configured dependencies for RTX 50-series GPUs
- ✅ WSL-optimized workflows

**Licensed under**: [MIT License](LICENSE.md)

---

## 🤝 Contributing & Community

Found a bug? Have an optimization? We welcome contributions!

1. **Report issues** with training, setup, or rendering
2. **Share improvements** – PRs for performance, UI, or documentation
3. **Share scenes** – Example datasets for reproducibility testing

**Discussions**:
- 💬 [GitHub Discussions](https://github.com/adityagit94/Gaussian-Splatting/discussions)
- 🐛 [Issue Tracker](https://github.com/adityagit94/Gaussian-Splatting/issues)

---

## 🔗 Quick Links & Resources

**Datasets**:
- [T&T+DB (COLMAP ready)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip) – 650 MB reference dataset
- [Pre-trained Models](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip) – 14 GB (for evaluation)
- [Evaluation Images](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/evaluation/images.zip) – 7 GB

**Viewers & Tools**:
- [SIBR Viewers (Windows)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/binaries/viewers.zip) – 60 MB OpenGL viewer

**Learning Resources**:
- 🎓 [Step-by-Step Tutorial](https://www.youtube.com/watch?v=UXtuigy_wYc) by Jonathan Stephens
- 📔 [Colab Notebook](https://github.com/camenduru/gaussian-splatting-colab) (quick testing)
- 📑 [Official Graphdeco Docs](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

---

## ⭐ Acknowledgments

- **Graphdeco-Inria** – Original 3D Gaussian Splatting research & implementation
- **Bernhard Kerbl et al.** – SIGGRAPH 2023 paper and foundational work
- **Community** – Feedback, datasets, and improvements from the research community

This project is maintained to ensure reproducibility and ease of use for the Windows/WSL community.

