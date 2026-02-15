import torch

from scene.gaussian_model import build_scaling_rotation
from utils.noise_steering_utils import (
    build_error_averager,
    compute_noise_scale,
    compute_per_pixel_error_map,
    normalize_noise_guidance,
    uses_error_map,
)


def _noise_option(opt, full_args, *names, default=None):
    for name in names:
        if hasattr(opt, name):
            value = getattr(opt, name)
            if value is not None:
                return value
        if hasattr(full_args, name):
            value = getattr(full_args, name)
            if value is not None:
                return value
    return default


def get_noise_error_downscale(opt):
    return max(1, int(getattr(opt, "noise_error_downscale", 1)))


class NoiseController:
    def __init__(self, opt, full_args):
        self.noise_guidance = normalize_noise_guidance(
            _noise_option(opt, full_args, "noise_guidance", default="opacity")
        )
        self.noise_percentile_threshold = float(
            _noise_option(opt, full_args, "noise_percentile_threshold", default=0.0)
        )
        self.noise_absolute_threshold = float(
            _noise_option(
                opt,
                full_args,
                "noise_absolute_threshold",
                "noise_error_absolute_threshold",
                default=0.005,
            )
        )
        self.noise_error_absolute_threshold = float(
            _noise_option(
                opt,
                full_args,
                "noise_error_absolute_threshold",
                "noise_absolute_threshold",
                default=self.noise_absolute_threshold,
            )
        )
        self.noise_amplification = float(
            _noise_option(opt, full_args, "noise_amplification", default=1.0)
        )
        self.per_pixel_error_metric = str(
            _noise_option(
                opt,
                full_args,
                "per_pixel_error_metric",
                "per_piexl_error_metric",
                default="l1",
            )
        ).lower()
        self.per_pixel_patch_size = int(
            _noise_option(opt, full_args, "per_pixel_patch_size", default=1)
        )
        avg_mode = _noise_option(
            opt, full_args, "noise_error_avg_mode", "error_averaging", default="windowed"
        )
        avg_window = int(
            _noise_option(
                opt,
                full_args,
                "noise_error_moving_average_window_size",
                "moving_average_window_size",
                default=100,
            )
        )
        avg_ema_decay = float(
            _noise_option(opt, full_args, "noise_error_ema_decay", "noise_ema", default=0.9)
        )
        self.use_error_guidance = uses_error_map(self.noise_guidance)
        self.error_avg = (
            build_error_averager(avg_mode, window_size=avg_window, ema_decay=avg_ema_decay)
            if self.use_error_guidance
            else None
        )
        self.fake_color = None
        self.error_contribution = None

    def _ensure_state(self, gaussians):
        if self.fake_color is None or self.fake_color.shape[0] != gaussians.get_xyz.shape[0]:
            self.fake_color = torch.zeros_like(gaussians.get_xyz, requires_grad=True, device="cuda")
            self.error_contribution = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
            if self.error_avg is not None:
                self.error_avg.reset()
        elif self.error_contribution is None or self.error_contribution.shape[0] != gaussians.get_xyz.shape[0]:
            self.error_contribution = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")

    def compute_fake_loss(self, image, gt_image, fake_viewpoint_cam, gaussians, pipe, render_fn):
        if not self.use_error_guidance:
            return None
        self._ensure_state(gaussians)
        black_bg = torch.zeros((3), device="cuda")
        fake_render = render_fn(
            fake_viewpoint_cam, gaussians, pipe, black_bg, override_color=self.fake_color
        )["render"]
        loss_per_pixel = compute_per_pixel_error_map(
            image.detach(),
            gt_image,
            metric=self.per_pixel_error_metric,
            patch_size=self.per_pixel_patch_size,
        )
        if fake_render.shape[-2:] != loss_per_pixel.shape:
            loss_per_pixel = torch.nn.functional.interpolate(
                loss_per_pixel.unsqueeze(0).unsqueeze(0),
                size=fake_render.shape[-2:],
                mode="area",
            ).squeeze(0).squeeze(0)
        return torch.sum((fake_render * loss_per_pixel).view(-1))

    def update_after_backward(self):
        if not self.use_error_guidance or self.fake_color is None or self.fake_color.grad is None:
            return
        with torch.no_grad():
            current_error = self.fake_color.grad[:, 0:1]
            if self.error_avg is not None:
                self.error_contribution = self.error_avg.update(current_error)
            else:
                self.error_contribution = current_error
            self.fake_color.grad = None

    def apply_noise(self, gaussians, args, xyz_lr, iteration, tb_writer=None):
        L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
        actual_covariance = L @ L.transpose(1, 2)
        if self.error_contribution is None or self.error_contribution.shape[0] != gaussians.get_xyz.shape[0]:
            self.error_contribution = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
        noise_scale, noise_threshold = compute_noise_scale(
            noise_guidance=self.noise_guidance,
            opacity=gaussians.get_opacity,
            importance_score=gaussians.importance_score,
            error_contribution=self.error_contribution,
            noise_percentile_threshold=self.noise_percentile_threshold,
            noise_absolute_threshold=self.noise_absolute_threshold,
            noise_error_absolute_threshold=self.noise_error_absolute_threshold,
            noise_amplification=self.noise_amplification,
        )
        if tb_writer and noise_threshold is not None:
            tb_writer.add_scalar("noise/noise_threshold", noise_threshold.item(), iteration)

        noise = torch.randn_like(gaussians._xyz) * noise_scale * args.noise_lr * xyz_lr
        noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
        gaussians._xyz.add_(noise)
