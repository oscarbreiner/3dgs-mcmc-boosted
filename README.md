# Improved MCMC-3DGS: Structure-Aware Noise and Aggressive Densification

**Extended Work Building on 3D Gaussian Splatting as Markov Chain Monte Carlo (NeurIPS 2024 Spotlight)**

### Our Team
- **Oscar Breiner**
- **Maximilian Leutschafft**

### Base Work
This project extends the work from [3DGS-MCMC](https://github.com/ubc-vision/3dgs-mcmc):

[![button](https://img.shields.io/badge/Original%20Project-orange?style=for-the-badge)](https://ubc-vision.github.io/3dgs-mcmc/)
[![button](https://img.shields.io/badge/Original%20Paper-blue?style=for-the-badge)](https://arxiv.org/abs/2404.09591)

**Original Authors:**
Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, Kwang Moo Yi

<hr>

## Project Overview

We build on 3DGS-MCMC, which uses **Stochastic Gradient Langevin Dynamics (SGLD)** to enable training from random initialization:

$$g \leftarrow g - \lambda_{\text{lr}} \cdot \nabla_g \mathbb{E}_{I}[L_{\text{total}}(g; I)] + \lambda_{\text{noise}} \cdot \varepsilon$$

Baseline noise depends only on opacity:

$$\varepsilon_\mu = \lambda_{\text{lr}} \cdot \sigma\big(-k(t - o)\big) \cdot \Sigma_\eta$$

This robustness comes with **slow convergence** due to unguided noisy exploration.

### 1️⃣ Structure-Aware Noise Steering

We scale noise based on the **per-Gaussian loss contribution**:

$$E_k = \sum_{u \in \text{Pix}} E(u) \cdot w_k(u)$$

Gaussians with higher $E_k$ receive more noise → prioritizing correction where reconstruction error is high.

We compare **L1 vs. SSIM** loss signals and **linear vs. sigmoid** noise mapping.

### 2️⃣ Aggressive Densification

We promote reliable structure by:

* **Pruning** low-opacity Gaussians
* **Splitting** high-importance Gaussians identified by:

$$i_{\max} = \arg\max_i w_i$$

Birth and death are **decoupled**, enabling faster structural growth while maintaining MCMC validity.

### Goal

> **Accelerate convergence** and **improve structure recovery** in sparse-view settings while preserving MCMC robustness.

<hr>

## Original 3DGS-MCMC Citation

```bibtex
@inproceedings{kheradmand20243d,
    title = {3D Gaussian Splatting as Markov Chain Monte Carlo},
    author = {Kheradmand, Shakiba and Rebain, Daniel and Sharma, Gopal and Sun, Weiwei and Tseng, Yang-Che and Isack, Hossam and Kar, Abhishek and Tagliasacchi, Andrea and Yi, Kwang Moo},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year = {2024},
    note = {Spotlight Presentation},
}
```

<hr>

## Updates

### Dec. 29th, 2024 — Extended Work
This fork introduces:
- **Structure-aware noise steering** based on per-Gaussian loss contributions
- **Aggressive densification** with decoupled birth/death operations
- Focus on **faster convergence** and **improved structure recovery** in sparse-view settings

### Dec. 5th, 2024 — Original 3DGS-MCMC Update
A new change has been pushed to diff-gaussian-rasterization. In order to pull it:
```sh
cd submodules/diff-gaussian-rasterization
git pull origin gs-mcmc
cd ../..
pip install submodules/diff-gaussian-rasterization
```

This change incorporates "Section B.2 Tighter Bounding of 2D Gaussians" from [StopThePop](https://arxiv.org/abs/2402.00525) paper. This bound allows to fit a tighter bound around Gaussians when opacity is less than 1.

## Installation

This project is built on top of the [Original 3DGS code base](https://github.com/graphdeco-inria/gaussian-splatting) and has been tested only on Ubuntu 20.04. If you encounter any issues, please refer to the [Original 3DGS code base](https://github.com/graphdeco-inria/gaussian-splatting) for installation instructions.

### Installation Steps

1. **Clone the Repository:**
   ```sh
   git clone --recursive https://github.com/oscarbreiner/3dgs-mcmc-boosted.git
   cd 3dgs-mcmc-boosted
   ```
2. **Set Up the Conda Environment:**
    ```sh
    conda create -y -n 3dgs-mcmc-env python=3.8
    conda activate 3dgs-mcmc-env
    ```
3. **Install Dependencies:**
    ```sh
    pip install plyfile tqdm torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    conda install cudatoolkit-dev=11.7 -c conda-forge
    ```
4. **Install Submodules:**
    ```sh
    CUDA_HOME=PATH/TO/CONDA/envs/3dgs-mcmc-env/pkgs/cuda-toolkit/ pip install submodules/diff-gaussian-rasterization submodules/simple-knn/
    ```
### Common Issues:
1. **Access Error During Cloning:**
If you encounter an access error when cloning the repository, ensure you have your SSH key set up correctly. Alternatively, you can clone using HTTPS.
2. **Running diff-gaussian-rasterization Fails:**
You may need to change the compiler options in the setup.py file to run both the original and this code. Update the setup.py with the following extra_compile_args:
    ```sh
    extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique", "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]}
    ```
    Afterwards, you need to reinstall diff-gaussian-rasterization. This is mentioned in [3DGS-issue-#41](https://github.com/graphdeco-inria/gaussian-splatting/issues/41).
    
By following these steps, you should be able to install the project and reproduce the results. If you encounter any issues, refer to the original 3DGS code base for further guidance.

## Usage

Running code is similar to the [Original 3DGS code base](https://github.com/graphdeco-inria/gaussian-splatting) with the following differences:
- You need to specify the maximum number of Gaussians that will be used. This is performed using --cap_max argument. The results in the paper uses the final number of Gaussians reached by the original 3DGS run for each shape.
- You need to specify the scale regularizer coefficient. This is performed using --scale_reg argument. For all the experiments in the paper, we use 0.01.
- You need to specify the opacity regularizer coefficient. This is performed using --opacity_reg argument. For Deep Blending dataset, we use 0.001. For all other experiments in the paper, we use 0.01.
- You need to specify the noise learning rate. This is performed using --noise_lr argument. For all the experiments in the paper, we use 5e5.
- You need to specify the initialization type. This is performed using --init_type argument. Options are random (to initialize randomly) or sfm (to initialize using a pointcloud).

### Basic Training Command
```sh
python train.py --source_path PATH/TO/Shape --config configs/shape.json --eval
```

### Extended Features (Coming Soon)
- Structure-aware noise steering with configurable loss signal (L1/SSIM)
- Aggressive densification strategies
- Enhanced sparse-view reconstruction




