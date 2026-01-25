# Training Metrics

This document describes the metrics currently logged during training, where they appear, and how to interpret them. Metrics are logged to W&B when enabled and to TensorBoard when available. Some metrics are only logged at specific intervals (see notes below).

## Logging scope

- W&B: logs every iteration unless otherwise noted.
- TensorBoard: logs key scalars every iteration, plus histograms at the densification interval.
- Test metrics: logged only at `test_iterations`.

## Core optimization metrics

- `total_loss` (W&B), `train_loss_patches/total_loss` (TB): Total training loss per iteration (photometric + regularizers). Lower is better.
- `photometric_loss` (W&B): Combined L1/SSIM photometric loss. Lower is better.
- `regularization_loss_opacity` (W&B): L1 penalty on opacity. Tracks opacity sparsity pressure.
- `regularization_loss_covariance` (W&B): L1 penalty on scaling. Tracks scale regularization pressure.
- `train_loss_patches/l1_loss` (TB): L1 loss per training patch.
- `SSIM` (W&B): Structural similarity on the current training view. Higher is better.
- `iter_time` (TB): Wall-clock time per iteration in ms.
- `time/iter_ms` (W&B): Wall-clock time per iteration in ms.
- `optim/xyz_lr` (W&B): Current positional learning rate after scheduling.

## Gaussian population and visibility

- `num_gaussians_total` (W&B): Total budget cap (from `cap_max`).
- `num_gaussians_total` (TB `total_points`): Current number of Gaussians in the model.
- `num_gaussians_alive` (W&B): Count of Gaussians with opacity > 0.005.
- `num_gaussians_dead` (W&B): Total minus alive count.
- `alive_ratio_percent` (W&B): Alive ratio in percent.
- `num_visible_gaussians` (W&B): Gaussians with non-zero screen radius in the current view.
- `num_max_contributor_gaussians` (W&B): Gaussians that were max-contributor for at least one pixel in the current view.
- `mean_blending_weight` (W&B): Mean opacity of visible Gaussians in the current view (proxy for average contribution weight).
- `util/visible_over_alive` (W&B): Ratio of visible Gaussians to alive Gaussians in the current view.
- `scene/opacity_histogram` (TB): Distribution of opacities at test iterations.

Interpretation:
- If `num_visible_gaussians` is low relative to total, many Gaussians are culled or small.
- If `alive_ratio_percent` is low, relocation or regularization may be too aggressive.

## Relocation proxy metrics

These reflect the chosen `reloc_sampling` strategy: `opacity`, `importance`, `error`, or `hybrid`.

Scalar proxy stats (W&B and TB):
- `proxy/mean`: Mean value of the active proxy.
- `proxy/median`: Median proxy value.
- `proxy/p95`: 95th percentile of proxy values.
- `proxy/zero_frac`: Fraction of proxy values that are zero.
- `proxy/entropy`: Entropy of the normalized proxy distribution (higher means more uniform).
- `proxy/top1pct_share`: Fraction of total proxy mass in the top 1% (higher means more concentrated).
- `proxy/corr_opacity`: Pearson correlation between proxy and opacity.
- `proxy/name` (W&B): The active proxy name.

Additional proxy correlation stats (enable with `--log_proxy_corr_all`):
- `proxy_corr/importance`: Correlation between importance and opacity.
- `proxy_corr/error`: Correlation between error and opacity.
- `proxy_corr/hybrid`: Correlation between hybrid and opacity.
- `proxy_corr/vis_opacity`: Correlation between visibility-weighted opacity and opacity.
- `proxy_corr/vis_importance`: Correlation between visibility-weighted importance and opacity.
- `proxy_corr/vis_hybrid`: Correlation between visibility-weighted hybrid and opacity.

Distributions and plots:
- `proxy/hist` (W&B): Histogram of proxy values, logged at the densification interval.
- `proxy/<name>` (TB): Histogram of proxy values, logged at the densification interval.
- `proxy/opacity_scatter` (W&B): Scatter of opacity vs proxy (sampled), logged at the densification interval when proxy is not opacity.

Interpretation:
- High `proxy/entropy` and low `proxy/top1pct_share` indicate broad sampling.
- High `proxy/corr_opacity` means the proxy is close to opacity; low means it is exploring different structure.
- A large `proxy/zero_frac` suggests sparse signals (common for importance/error early on).

Quick reading guide:
- The proxy is the per-Gaussian sampling weight used for relocation; logs summarize its distribution.
- `proxy/mean`, `proxy/median`, `proxy/p95` describe the proxy scale and tail.
- `proxy/zero_frac` shows how many Gaussians have no sampling weight.
- `proxy/entropy` and `proxy/top1pct_share` tell you how concentrated sampling is.
- `proxy/corr_opacity` tells you how similar the proxy is to opacity (near 1 means similar).
- `proxy/opacity_scatter` is a visual check of how proxy relates to opacity when they differ.

## Relocation dynamics

Logged at the densification interval:
- `reloc/num_dead`: Number of Gaussians considered dead (opacity <= 0.005).
- `reloc/num_relocated`: Number of Gaussians relocated from dead slots.
- `reloc/mean_target_prob`: Mean sampling probability of relocation targets.
- `reloc/num_added`: Number of new Gaussians added to reach the cap schedule.
- `reloc/mean_source_prob`: Mean sampling probability of sources for newly added Gaussians.
- `reloc/delta_photometric_loss` (W&B, TB): Change in photometric loss around relocation steps (logged on the next iteration, aligned to the relocation step).

Interpretation:
- If `reloc/num_dead` is high and persistent, consider adjusting regularization or relocation policy.
- Low `reloc/mean_target_prob` indicates sampling is spread out; high indicates concentration on a few Gaussians.

## Evaluation metrics (test iterations only)

- `test/loss_viewpoint - l1_loss` (TB): L1 loss on test views.
- `test/loss_viewpoint - psnr` (TB): PSNR on test views.
- `train/loss_viewpoint - l1_loss` (TB): L1 loss on sampled train views.
- `train/loss_viewpoint - psnr` (TB): PSNR on sampled train views.
- `eval/time_to_threshold_iter` (W&B, TB): First iteration where test PSNR crosses `--psnr_threshold`.
- `eval/time_to_threshold_s` (W&B, TB): Wall time in seconds to reach `--psnr_threshold`.
- `eval/psnr_threshold` (W&B, TB): The configured threshold used for time-to-threshold logging.

Logged only when `--psnr_threshold` > 0 and when test evaluation runs.

Example threshold: for the bicycle scene, the established baseline final test PSNR is 26.07. Using a 90% target, set `--psnr_threshold 23.46` to track time-to-23.46.

Interpretation:
- Use PSNR/L1 trends to compare runs. Divergence between train and test indicates overfitting.

## Logging cadence notes

- Scalars log every iteration.
- Histograms and scatter plots log at the densification interval (`densification_interval`).
- Test metrics log at `test_iterations`.
