# Hyperparameters and CLI options (current code)

This file mirrors the current defaults in `arguments/__init__.py`, `train.py`, and `render.py`.

## Configuration precedence (`train.py`)

1. Parser defaults are loaded first.
2. If `--config path.json` is set, JSON keys overwrite defaults for keys not explicitly passed on CLI.
3. Explicit CLI flags always win.

Special case: if `test_iterations` is not set in CLI or config, it is auto-filled as `0, 5000, 10000, ... , iterations`.

## Model parameters (`ModelParams`)

| Flag | Default | Notes |
| --- | --- | --- |
| `--sh_degree` | `3` | Max SH degree. |
| `--source_path`, `-s` | `""` | Dataset root. |
| `--model_path`, `-m` | `""` | Output directory. |
| `--images`, `-i` | `"images"` | Image subfolder name. |
| `--resolution`, `-r` | `-1` | `-1` keeps original resolution. |
| `--white_background`, `-w` | `False` | Use white background. |
| `--data_device` | `"cuda"` | Tensor device for data. |
| `--eval` | `False` | Dataset eval mode. |
| `--cap_max` | `-1` | Max Gaussian count; required by this repo at runtime. |
| `--init_type` | `"random"` | Dataset init mode (`random`, `sfm`, dataset-dependent extras). |

## Pipeline parameters (`PipelineParams`)

| Flag | Default | Notes |
| --- | --- | --- |
| `--convert_SHs_python` | `False` | Python SH conversion instead of CUDA path. |
| `--compute_cov3D_python` | `False` | Python covariance path. |
| `--debug` | `False` | Internal render debug flag. |

## Optimization parameters (`OptimizationParams`)

| Flag | Default | Notes |
| --- | --- | --- |
| `--iterations` | `30000` | Total train iterations. |
| `--position_lr_init` | `0.00016` | XYZ LR schedule start. |
| `--position_lr_final` | `0.0000016` | XYZ LR schedule end. |
| `--position_lr_delay_mult` | `0.01` | XYZ LR delay multiplier. |
| `--position_lr_max_steps` | `30000` | XYZ LR schedule length. |
| `--feature_lr` | `0.0025` | Feature LR. |
| `--opacity_lr` | `0.05` | Opacity LR. |
| `--scaling_lr` | `0.005` | Scale LR. |
| `--rotation_lr` | `0.001` | Rotation LR. |
| `--percent_dense` | `0.01` | Densification ratio parameter. |
| `--lambda_dssim` | `0.2` | SSIM mixing term in photometric loss. |
| `--densification_interval` | `100` | Densification cadence. |
| `--opacity_reset_interval` | `3000` | Opacity reset cadence. |
| `--densify_from_iter` | `500` | Densification start iteration. |
| `--densify_until_iter` | `25000` | Densification stop iteration. |
| `--densify_grad_threshold` | `0.0002` | Densification gradient threshold. |
| `--random_background` | `False` | Random background per iteration. |
| `--noise_lr` | `500000.0` | Noise multiplier term. |
| `--noise_guidance` | `"opacity"` | Supported: `opacity`, `error`, `opacity_error_percentile`, `opacity_error_threshold`, `random`. |
| `--noise_amplification` | `1.0` | Extra multiplier on noise scale. |
| `--noise_error_percentile_threshold` | `0.0` | Percentile threshold for percentile guidance modes. |
| `--noise_error_absolute_threshold` | `0.005` | Absolute threshold for threshold guidance modes. |
| `--noise_error_moving_average_window_size` | `100` | Window for moving-average error guidance. |
| `--noise_error_avg_mode` | `"windowed_moving_average"` | `windowed_moving_average`, `ema`, `none`. |
| `--noise_error_ema_decay` | `0.9` | EMA decay when `noise_error_avg_mode=ema`. |
| `--noise_error_downscale` | `1` | Error-guidance render downscale factor. |
| `--per_pixel_error_metric` | `"l1"` | Per-pixel error metric (`l1` or `psnr` path). |
| `--per_pixel_patch_size` | `1` | Patch size for local error smoothing. |
| `--scale_reg` | `0.01` | Scale regularization weight. |
| `--opacity_reg` | `0.01` | Opacity regularization weight. |
| `--reloc_sampling` | `"opacity"` | Supported: `random`, `opacity`, `vis_binary`, `vis_pixel_count`, `vis_pixel_count_snapshot`, `vis_pixel_count_ema_quantile`, `error`, `vis_pixel_count_hybrid`, `vis_binary_opacity`, `vis_binary_vis_pixel_count`, `vis_binary_vis_pixel_count_hybrid`. |
| `--vis_pixel_count_ema` | `0.9` | EMA decay for vis-pixel-count proxy. |
| `--error_ema` | `0.9` | EMA decay for error proxy. |
| `--vis_binary_ema` | `0.9` | EMA decay for binary-visibility proxy. |
| `--vis_pixel_count_mode` | `"count"` | `count` supported (`wsum` currently raises). |
| `--vis_pixel_count_snapshot_top_frac` | `0.01` | Top fraction kept for snapshot proxy. |
| `--vis_pixel_count_ema_quantile_top_frac` | `0.01` | Top fraction kept for EMA-quantile proxy. |
| `--vis_pixel_count_update_interval` | `1` | Update cadence for vis-pixel-count proxy. |
| `--vis_pixel_count_subsample_stride` | `1` | Spatial stride before counting max-contributor IDs. |
| `--vis_pixel_count_subsample_ratio` | `1.0` | Random ratio after stride subsampling. |
| `--log_proxy_corr_all` | `False` | Log full proxy correlation set. |
| `--correlation_analysis` | `False` | Enables correlation CSV + extra correlation logs. |
| `--cleanup_random_ply` | `True` | Remove generated random init PLY after training. |
| `--psnr_threshold` | `23.46` | Logs time-to-threshold when test PSNR crosses this. |

## Train runtime flags (`train.py`)

| Flag | Default | Notes |
| --- | --- | --- |
| `--config` | `None` | JSON config file path. |
| `--debug_from` | `-1` | Turns on pipeline debug at/after this iteration. |
| `--detect_anomaly` | `False` | PyTorch anomaly detection. |
| `--test_iterations` | parser: `[7000, 30000]` | Effective default is auto-generated schedule described above. |
| `--save_iterations` | `[7000, 30000]` | `iterations` is always appended at runtime. |
| `--quiet` | `False` | Reduce console noise. |
| `--checkpoint_iterations` | `[]` | Extra checkpoint saves. |
| `--start_checkpoint` | `None` | Resume checkpoint path. |
| `--wandb_project` | `None` | Override W&B project. |
| `--wandb_run_name` | `None` | Override run name. |
| `--wandb_run_group` | `None` | Optional W&B group. |
| `--scene_id` | `None` | Optional run metadata. |
| `--scene_type` | `None` | Optional run metadata. |
| `--mlflow` | `False` | Explicit MLflow enable (W&B path can also enable MLflow). |
| `--logging_level` | `"core"` | `core`, `diagnostic`, `analysis`. |

## Render flags (`render.py`)

`render.py` also accepts `ModelParams` + `PipelineParams` flags above.

| Flag | Default | Notes |
| --- | --- | --- |
| `--iteration` | `-1` | `-1` loads latest available iteration. |
| `--skip_train` | `False` | Skip rendering train split. |
| `--skip_test` | `False` | Skip rendering test split. |
| `--quiet` | `False` | Reduce console output. |
