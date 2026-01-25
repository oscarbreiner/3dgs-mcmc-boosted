# Hyperparameters and CLI options

This project exposes training, rendering, and evaluation settings through CLI flags and (optionally) JSON config files. Defaults listed below are the in-code defaults.

## Configuration precedence

Training (`train.py`) supports `--config` (JSON). Any CLI flag you provide overrides the config value; otherwise the config value replaces the default.

Rendering (`render.py`) reads `cfg_args` from the `--model_path` directory (written during training) and then applies CLI overrides.

## Loading / model parameters (ModelParams)

| Flag | Type / options | Default | What it does |
| --- | --- | --- | --- |
| `--sh_degree` | int | `3` | Max spherical harmonics degree (increases over time during training). |
| `--source_path`, `-s` | path | `""` | Dataset root path. |
| `--model_path`, `-m` | path | `""` | Output folder for checkpoints and logs. |
| `--images`, `-i` | str | `"images"` | Subfolder name containing input images. |
| `--resolution`, `-r` | int | `-1` | Image resolution; `-1` keeps dataset resolution. |
| `--white_background`, `-w` | bool | `False` | Use white background instead of black. |
| `--data_device` | str | `"cuda"` | Device for dataset tensors (e.g., `cuda`, `cpu`). |
| `--eval` | bool | `False` | Load dataset in evaluation mode (e.g., LLFF holdout). |
| `--cap_max` | int | `-1` | Max number of Gaussians; required by this repo (see README). |
| `--init_type` | str | `"random"` | Initialization mode: `random` or `sfm`. |

## Pipeline parameters (PipelineParams)

| Flag | Type / options | Default | What it does |
| --- | --- | --- | --- |
| `--convert_SHs_python` | bool | `False` | Use Python SH conversion instead of CUDA. |
| `--compute_cov3D_python` | bool | `False` | Use Python 3D covariance instead of CUDA. |
| `--debug` | bool | `False` | Enable pipeline debug in render loop. |

## Optimization parameters (OptimizationParams)

| Flag | Type / options | Default | What it does |
| --- | --- | --- | --- |
| `--iterations` | int | `30000` | Total training iterations. |
| `--position_lr_init` | float | `0.00016` | Initial position learning rate. |
| `--position_lr_final` | float | `0.0000016` | Final position learning rate. |
| `--position_lr_delay_mult` | float | `0.01` | Delay multiplier for position LR warmup. |
| `--position_lr_max_steps` | int | `30000` | Steps for position LR schedule. |
| `--feature_lr` | float | `0.0025` | Feature learning rate. |
| `--opacity_lr` | float | `0.05` | Opacity learning rate. |
| `--scaling_lr` | float | `0.005` | Scaling learning rate. |
| `--rotation_lr` | float | `0.001` | Rotation learning rate. |
| `--percent_dense` | float | `0.01` | Percent of densified points per interval. |
| `--lambda_dssim` | float | `0.2` | DSSIM weight in photometric loss. |
| `--densification_interval` | int | `100` | Iterations between densification checks. |
| `--opacity_reset_interval` | int | `3000` | Iterations between opacity resets. |
| `--densify_from_iter` | int | `500` | Start densification at this iteration. |
| `--densify_until_iter` | int | `25000` | Stop densification at this iteration. |
| `--densify_grad_threshold` | float | `0.0002` | Gradient threshold to trigger densification. |
| `--random_background` | bool | `False` | Use random background colors during training. |
| `--noise_lr` | float | `5e5` | Noise injection scale for relocation. |
| `--scale_reg` | float | `0.01` | Scaling regularizer weight. |
| `--opacity_reg` | float | `0.01` | Opacity regularizer weight. |
| `--reloc_sampling` | str | `"opacity"` | Relocation sampling strategy: `opacity`, `random`, `importance`, `error`, `hybrid`, `vis_opacity`, `vis_importance`, `vis_hybrid`. |
| `--importance_ema` | float | `0.9` | EMA for importance proxy updates. |
| `--error_ema` | float | `0.9` | EMA for error proxy updates. |
| `--visibility_ema` | float | `0.9` | EMA for visibility proxy updates. |
| `--importance_mode` | str | `"count"` | Importance proxy mode: `count` (supported), `wsum` (not supported). |
| `--log_proxy_corr_all` | bool | `False` | Log proxy correlation metrics each interval. |
| `--psnr_threshold` | float | `-1.0` | PSNR target used to log time-to-threshold; set `-1` to disable. |

## Training runtime parameters (train.py)

| Flag | Type / options | Default | What it does |
| --- | --- | --- | --- |
| `--config` | path | `None` | Optional JSON config file. |
| `--debug_from` | int | `-1` | Enable render pipeline debug from this iteration. |
| `--detect_anomaly` | bool | `False` | Enable autograd anomaly detection. |
| `--test_iterations` | list[int] | `7000 30000` | Iterations to run evaluation renders. |
| `--save_iterations` | list[int] | `7000 30000` | Iterations to save checkpoints; final iteration is appended. |
| `--checkpoint_iterations` | list[int] | `[]` | Iterations to write extra checkpoints. |
| `--start_checkpoint` | path | `None` | Restore from checkpoint. |
| `--quiet` | bool | `False` | Reduce console output. |

## Rendering parameters (render.py)

| Flag | Type / options | Default | What it does |
| --- | --- | --- | --- |
| `--iteration` | int | `-1` | Which training iteration to render (`-1` = latest). |
| `--skip_train` | bool | `False` | Skip train split rendering. |
| `--skip_test` | bool | `False` | Skip test split rendering. |
| `--quiet` | bool | `False` | Reduce console output. |
