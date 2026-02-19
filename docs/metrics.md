# Training metrics (current code)

This file lists the metric keys currently emitted by `train.py` and `trainer/eval.py`.

## Backends and cadence

- W&B: logged only if W&B is installed and a run is active.
- TensorBoard: logged only if `torch.utils.tensorboard` is available.
- MLflow: mirrors numeric logs through `trainer/logging.py`.
- Evaluation metrics: only at `test_iterations`.

## Per-iteration core metrics (W&B)

- `loss/total`
- `loss/photometric`
- `loss/reg_opacity`
- `loss/reg_covariance`
- `loss/train_l1`
- `quality/train_ssim`
- `quality/train_psnr`
- `perf/iter_ms`
- `optim/xyz_lr`
- `pop/num_alive`
- `pop/num_dead`
- `pop/alive_ratio_percent`
- `vis/mean_blending_weight`
- `vis/num_max_contributor`
- `vis/num_visible`
- `util/visible_over_alive`

## Per-iteration TensorBoard metrics

- `train_loss_patches/l1_loss`
- `train_loss_patches/total_loss`
- `train_loss_patches/psnr`
- `iter_time`

## Logging level gated metrics

`--logging_level` controls extra proxy logging:

- `core`: no proxy stats.
- `diagnostic`: proxy distribution stats.
- `analysis`: diagnostic + cross-proxy correlation stats (when enabled).

### Diagnostic proxy stats (W&B + TB)

- `proxy/name` (W&B)
- `proxy/mean`
- `proxy/median`
- `proxy/p95`
- `proxy/zero_frac`
- `proxy/entropy`
- `proxy/top1pct_share`
- `proxy/corr_opacity`

Proxy histograms:

- TB `proxy/<proxy_name>` every `max(1, densification_interval)` iterations (only when a proxy exists for selected relocation mode).

Proxy scatter (W&B, analysis mode only):

- `proxy/opacity_scatter` sampled scatter, every 10k iterations (gated by internal interval logic).

### Full proxy correlations (`--log_proxy_corr_all`, analysis mode)

- W&B and TB: `proxy_corr/<name>` for:
- `vis_pixel_count`
- `vis_pixel_count_snapshot`
- `vis_pixel_count_ema_quantile`
- `error`
- `vis_pixel_count_hybrid`
- `vis_binary`
- `vis_binary_opacity`
- `vis_binary_vis_pixel_count`
- `vis_binary_vis_pixel_count_hybrid`

## Relocation / densification metrics

Logged at densification steps (`iteration % densification_interval == 0` within densification window):

- `reloc/num_dead`
- `reloc/num_relocated`
- `reloc/mean_target_prob`
- `reloc/num_added`
- `reloc/mean_source_prob`

`reloc/delta_photometric_loss` is logged one iteration later but written at the relocation step index.

## Noise metric (TensorBoard)

- `noise/noise_threshold` when the active noise guidance mode returns a threshold (for percentile-based paths).

## Evaluation metrics (`test_iterations` only)

W&B/MLflow:

- `eval/test/l1`, `eval/test/psnr`, `eval/test/ssim`
- `eval/test/lpips` (only if LPIPS package is installed)
- `eval/train/l1`, `eval/train/psnr`

TensorBoard:

- `test/loss_viewpoint - l1_loss`
- `test/loss_viewpoint - psnr`
- `test/loss_viewpoint - ssim`
- `test/loss_viewpoint - lpips` (if LPIPS available)
- `train/loss_viewpoint - l1_loss`
- `train/loss_viewpoint - psnr`
- `scene/opacity_histogram`
- `total_points`

Best-value summaries (W&B summary + MLflow):

- `eval/test/psnr_best`, `eval/test/psnr_best_iter`
- `eval/test/l1_best`, `eval/test/l1_best_iter`
- `eval/train/psnr_best`, `eval/train/psnr_best_iter`
- `eval/train/l1_best`, `eval/train/l1_best_iter`

## Time-to-threshold metrics

When `psnr_threshold > 0` and first crossed on a test eval:

- `eval/time_to_threshold_iter`
- `eval/time_to_threshold_s`
- `eval/psnr_threshold`

## End-of-training metrics

At final iteration:

- `train/total_time_s`
- `pop/num_final`
- `pop/num_final_million`
- `pop/num_final_alive`

## Correlation analysis extras (`--correlation_analysis`)

- W&B `corr/<a>_<b>` entries for pairwise correlations among `opacity`, `vis_pixel_count`, `vis_binary` every 300 iterations.
- CSV dump at `<model_path>/correlation_scatter.csv` with sampled per-Gaussian rows:
- `step,gaussian_idx,opacity,vis_pixel_count,vis_binary,error,raw_vis_pixel_count,raw_vis_binary`
