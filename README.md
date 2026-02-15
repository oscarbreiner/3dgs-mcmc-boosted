# 3DGS-MCMC-Boosted: Targeted Exploration & Visibility-Based Relocation

This repository builds on **3D Gaussian Splatting as Markov Chain Monte Carlo (MCMC-3DGS, NeurIPS 2024 Spotlight)** and studies simple modifications that **speed up convergence** while keeping the **robustness to initialization** of the MCMC formulation.

**Team**
- Oscar Breiner
- Maximilian Leutschafft

**Base project:** https://github.com/ubc-vision/3dgs-mcmc  
**Paper:** https://arxiv.org/abs/2404.09591

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

### Common issue: diff-gaussian-rasterization compile flags
If building fails due to compiler/symbol issues, you may need to adjust `extra_compile_args` in the rasterizer `setup.py` (see gaussian-splatting issue #41), then reinstall the submodule.

---

## Usage

Training is similar to the upstream Gaussian Splatting / MCMC-3DGS workflow.

Example:
```sh
python train.py --source_path PATH/TO/SCENE --config configs/scene.json --eval
```

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

### Extended Features (Coming Soon)
- Structure-aware noise steering with configurable loss signal (L1/SSIM)
- Aggressive densification strategies
- Enhanced sparse-view reconstruction

## Recommended Setup

Current best setup in this repo:
- **Noise steering:** hybrid opacity + error threshold (`--noise_guidance opacity_error_threshold`)
- **Relocation sampling:** `--reloc_sampling vis_pixel_count`
