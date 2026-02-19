# 3DGS-MCMC-Boosted: Targeted Exploration & Visibility-Based Relocation

This repository builds on **3D Gaussian Splatting as Markov Chain Monte Carlo (MCMC-3DGS, NeurIPS 2024 Spotlight)** and investigates targeted noise exploration, visibility-aware relocation, and renderer-level instrumentation to better understand and improve training behavior across scenes.

**Team**
- Oscar Breiner
- Maximilian Leutschafft

**Base project:** https://github.com/ubc-vision/3dgs-mcmc  
**Base Paper:** https://arxiv.org/abs/2404.09591

---

## Background (Short)

### 3D Gaussian Splatting (3DGS)
3DGS represents a scene as many anisotropic 3D Gaussians (position, covariance, opacity, color) and renders them by projecting to the image plane and alpha-compositing. Training minimizes a photometric loss. Classical 3DGS typically increases capacity with heuristic **densification** (clone/split high-gradient Gaussians) and removes low-contributing ones via **pruning**. It works well with good SfM initialization, but can be unstable and strongly initialization-dependent.

### MCMC-based 3DGS
MCMC-3DGS interprets training as sampling from an implicit distribution proportional to rendering quality and updates Gaussian parameters using **SGLD** (gradient step + injected noise). This enables exploration and reduces reliance on SfM. Instead of clone/split densification, it uses **relocation**: “dead” (low-opacity) Gaussians are moved onto “live” ones with parameter adjustments designed to approximately preserve the rendered image, keeping training stable.

---

## Our Idea: Purposeful Noise + Smarter Capacity Allocation

We aim to keep MCMC-3DGS robustness, but reduce wasted randomness and place capacity where it matters:

- **Opacity + error-gated noise steering:** inject stronger noise into Gaussians that are both *eligible* (opacity-gated) and associated with **high residual reconstruction error**, encouraging exploration where the model underfits.
- **Visibility-based relocation:** when relocating dead Gaussians, choose targets using **visibility/importance** signals (not just opacity), so new capacity is allocated to Gaussians that are structurally useful and actually observed in views.

---

## Citation (Original MCMC-3DGS)
```bibtex
@inproceedings{kheradmand20243d,
    title = {3D Gaussian Splatting as Markov Chain Monte Carlo},
    author = {Kheradmand, Shakiba and Rebain, Daniel and Sharma, Gopal and Sun, Weiwei and Tseng, Yang-Che and Isack, Hossam and Kar, Abhishek and Tagliasacchi, Andrea and Yi, Kwang Moo},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year = {2024},
    note = {Spotlight Presentation},
}
```

---

## Installation

This project is based on the original Gaussian Splatting codebase and the MCMC-3DGS fork. We tested on **Ubuntu 20.04**.

1) Clone:
```sh
git clone --recursive https://github.com/oscarbreiner/3dgs-mcmc-boosted.git
cd 3dgs-mcmc-boosted
```

2) Create environment:
```sh
conda create -y -n 3dgs-mcmc-env python=3.8
conda activate 3dgs-mcmc-env
```

3) Install PyTorch (example for CUDA 11.7):
```sh
pip install plyfile tqdm
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
conda install cudatoolkit-dev=11.7 -c conda-forge
```

4) Install submodules:
```sh
CUDA_HOME=PATH/TO/CONDA/envs/3dgs-mcmc-env/pkgs/cuda-toolkit/ \
  pip install submodules/diff-gaussian-rasterization submodules/simple-knn/
```

Rasterizer fork used in this project:
- https://github.com/oscarbreiner/diff-gaussian-rasterization.git

### Common Issues
1. Access error during cloning:
If cloning via SSH fails, verify your SSH key setup and repository access. As an alternative, clone via HTTPS.

2. `diff-gaussian-rasterization` build fails:
If compilation fails (often due to compiler/symbol settings), update `extra_compile_args` in the rasterizer `setup.py`:

```sh
extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique", "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]}
```

Then reinstall `diff-gaussian-rasterization`.
Reference: https://github.com/graphdeco-inria/gaussian-splatting/issues/41

---

## Usage

Training is similar to the upstream Gaussian Splatting / MCMC-3DGS workflow.

Example:
```sh
python train.py --source_path PATH/TO/SCENE --config configs/scene.json --eval
```

## What Was Added In This Project

Noise exploration controls:
- `noise_guidance` modes: `opacity`, `error`, `opacity_error_percentile`, `opacity_error_threshold`, `random`
- Noise scale and gating knobs: `noise_lr`, `noise_amplification`, `noise_error_percentile_threshold`, `noise_error_absolute_threshold`
- Error-map and smoothing knobs: `per_pixel_error_metric` (`l1` or `psnr` path), `per_pixel_patch_size`, `noise_error_avg_mode` (`windowed_moving_average`, `ema`, `none`), `noise_error_moving_average_window_size`, `noise_error_ema_decay`
- Optional reduced-resolution error-guidance pass: `noise_error_downscale`

Relocation and sampling controls:
- `reloc_sampling` modes: `random`, `opacity`, `vis_binary`, `vis_pixel_count`, `vis_pixel_count_snapshot`, `vis_pixel_count_ema_quantile`, `error`, `vis_pixel_count_hybrid`, `vis_binary_opacity`, `vis_binary_vis_pixel_count`, `vis_binary_vis_pixel_count_hybrid`
- Proxy update controls: `vis_pixel_count_ema`, `error_ema`, `vis_binary_ema`, `vis_pixel_count_mode` (`count`; `wsum` currently unsupported), `vis_pixel_count_snapshot_top_frac`, `vis_pixel_count_ema_quantile_top_frac`, `vis_pixel_count_update_interval`, `vis_pixel_count_subsample_stride`, `vis_pixel_count_subsample_ratio`
- Analysis/logging controls: `logging_level` (`core`, `diagnostic`, `analysis`), `log_proxy_corr_all`, `correlation_analysis`

Renderer and training instrumentation:
- Training consumes additional rasterizer outputs for visibility/attribution proxies (`max_id`, visibility masks).
- Expanded logging for relocation/proxy dynamics (W&B / TensorBoard / MLflow).

ScanNet++ support:
- Native scene detection/loading in the main training pipeline (`scene/__init__.py`, `scene/dataset_readers.py`).
- No runtime import from `scannetpp/` is required by `train.py`.

## Experiment Tracking

This project supports both **W&B** (recommended for SLURM) and **TensorBoard**.

### W&B Setup (Recommended)

```bash
# One-time login
wandb login

# Or set API key in SLURM script
export WANDB_API_KEY=your_api_key_here
```

### TensorBoard (Alternative)

```bash
tensorboard --logdir=output/your_run_directory
```

Requires SSH port forwarding: `ssh -L 6006:localhost:6006 your_cluster`

## Recommended Setup

Current best setup in this repo:
- **Noise steering:** hybrid opacity + error threshold (`--noise_guidance opacity_error_threshold`)
- **Relocation sampling:** `--reloc_sampling vis_pixel_count`
