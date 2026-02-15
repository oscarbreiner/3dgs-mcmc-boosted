#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import json
import time
import math
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.noise_steering_utils import (
    build_error_averager,
    compute_noise_scale,
    compute_per_pixel_error_map,
    normalize_noise_guidance,
    uses_error_map,
)
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.gaussian_model import build_scaling_rotation
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False
try:
    import mlflow
    MLFLOW_FOUND = True
except ImportError:
    MLFLOW_FOUND = False
try:
    from lpipsPyTorch import LPIPS
    LPIPS_FOUND = True
except ImportError:
    LPIPS_FOUND = False

_LPIPS_NET_TYPE = "vgg"
_LPIPS_MODEL = None
_MLFLOW_ACTIVE = False
_MLFLOW_ENABLED = False

def _mlflow_log_params_safe(params):
    if not _MLFLOW_ACTIVE:
        return
    safe_params = {}
    for key, value in params.items():
        if isinstance(value, (list, tuple, dict)):
            safe_params[key] = json.dumps(value)
        else:
            safe_params[key] = str(value)
    chunk = {}
    for key, value in safe_params.items():
        chunk[key] = value
        if len(chunk) >= 100:
            mlflow.log_params(chunk)
            chunk = {}
    if chunk:
        mlflow.log_params(chunk)

def _mlflow_log_metrics(metrics, step=None):
    if not _MLFLOW_ACTIVE:
        return
    safe_metrics = {}
    for key, value in metrics.items():
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        safe_metrics[key] = numeric
    if safe_metrics:
        mlflow.log_metrics(safe_metrics, step=step)

def _mlflow_set_tag_safe(key, value):
    if not _MLFLOW_ACTIVE:
        return
    if value is None:
        return
    mlflow.set_tag(str(key), str(value))

def _mlflow_log_image(name, image_np, iteration):
    if not _MLFLOW_ACTIVE:
        return
    log_image = getattr(mlflow, "log_image", None)
    if log_image is None:
        return
    artifact_name = "{}/{}.png".format(int(iteration), name)
    log_image(image_np, artifact_file=artifact_name)

def _mlflow_start_run(args, project_name, mlflow_enabled):
    global _MLFLOW_ACTIVE
    global _MLFLOW_ENABLED
    _MLFLOW_ENABLED = bool(mlflow_enabled)
    if not _MLFLOW_ENABLED:
        return False
    if not MLFLOW_FOUND:
        print("MLflow not installed: skipping MLflow logging")
        return False
    if _MLFLOW_ACTIVE:
        return True
    mlflow.set_experiment(project_name)
    run_name = getattr(args, "wandb_run_name", None) or getattr(args, "run_name", None)
    mlflow.start_run(run_name=run_name)
    tags = {
        "model_path": str(getattr(args, "model_path", "")),
        "reloc_sampling": str(getattr(args, "reloc_sampling", "")),
        "cap_max": str(getattr(args, "cap_max", "")),
        "correlation_analysis": str(bool(getattr(args, "correlation_analysis", False))),
    }
    mlflow.set_tags(tags)
    _mlflow_log_params_safe(vars(args))
    cfg_path = os.path.join(args.model_path, "cfg_args")
    if os.path.exists(cfg_path):
        mlflow.log_artifact(cfg_path)
    print("MLflow logging enabled")
    _MLFLOW_ACTIVE = True
    return True

def _get_lpips_model(device):
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None or next(_LPIPS_MODEL.parameters()).device != device:
        _LPIPS_MODEL = LPIPS(net_type=_LPIPS_NET_TYPE).to(device)
        _LPIPS_MODEL.eval()
    return _LPIPS_MODEL

def _safe_corrcoef(x, y):
    if x.numel() < 2 or y.numel() < 2:
        return float("nan")
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum()) * torch.sqrt((y * y).sum())
    if denom <= torch.finfo(x.dtype).eps:
        return float("nan")
    return float((x * y).sum().item() / denom.item())

def _proxy_stats(proxy_vals, opacity_vals):
    stats = {}
    proxy_vals = proxy_vals.float()
    opacity_vals = opacity_vals.float()
    if proxy_vals.numel() == 0:
        return stats
    stats["mean"] = float(proxy_vals.mean().item())
    stats["median"] = float(torch.quantile(proxy_vals, 0.5).item())
    stats["p95"] = float(torch.quantile(proxy_vals, 0.95).item())
    stats["zero_frac"] = float((proxy_vals <= 0).float().mean().item())
    p = torch.clamp(proxy_vals, min=0)
    s = p.sum()
    if s > torch.finfo(p.dtype).eps:
        p = p / s
        stats["entropy"] = float((-(p * torch.log(p + torch.finfo(p.dtype).eps))).sum().item())
        k = max(1, int(0.01 * p.numel()))
        stats["top1pct_share"] = float(torch.topk(p, k=k).values.sum().item())
    else:
        stats["entropy"] = float("nan")
        stats["top1pct_share"] = 0.0
    stats["corr_opacity"] = _safe_corrcoef(proxy_vals, opacity_vals)
    return stats

def _all_proxy_values(gaussians, opacity_vals):
    eps = torch.finfo(opacity_vals.dtype).eps
    visibility = torch.clamp(gaussians.visibility_score.to(dtype=opacity_vals.dtype), min=eps)
    return {
        "importance": gaussians.importance_score,
        "importance_snapshot": gaussians.importance_snapshot_score,
        "importance_ema_quantile": gaussians.get_importance_ema_quantile(),
        "error": gaussians.error_score,
        "hybrid": opacity_vals * gaussians.importance_score,
        "visibility": gaussians.visibility_score,
        "vis_opacity": opacity_vals * visibility,
        "vis_importance": gaussians.importance_score * visibility,
        "vis_hybrid": opacity_vals * gaussians.importance_score * visibility,
    }

def _proxy_corrs(opacity_vals, proxy_map):
    corrs = {}
    for name, proxy_vals in proxy_map.items():
        if proxy_vals is None or proxy_vals.numel() != opacity_vals.numel():
            continue
        corrs[name] = _safe_corrcoef(proxy_vals, opacity_vals)
    return corrs

def _subsample(tensor, max_samples):
    if tensor.numel() <= max_samples:
        return tensor
    idx = torch.randperm(tensor.numel(), device=tensor.device)[:max_samples]
    return tensor[idx]

def _subsample_max_id(max_id, stride=1, ratio=1.0):
    if max_id is None or max_id.numel() == 0:
        return max_id, 1.0
    stride = max(1, int(stride))
    ratio = float(ratio)
    ratio = min(max(ratio, 0.0), 1.0)
    total = max_id.numel()
    sampled = max_id
    if stride > 1 and sampled.dim() >= 2:
        sampled = sampled[::stride, ::stride]
    if ratio < 1.0:
        flat = sampled.reshape(-1)
        sample_size = max(1, int(ratio * flat.numel()))
        idx = torch.randperm(flat.numel(), device=flat.device)[:sample_size]
        sampled = flat[idx]
    sampled_numel = sampled.numel()
    if sampled_numel == 0:
        return sampled, 1.0
    scale = float(total) / float(sampled_numel)
    return sampled, scale

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

def _append_corr_scatter_rows(
    path,
    iteration,
    gaussians,
    raw_visibility=None,
    raw_importance=None,
    sample_frac=0.01,
):
    opacity = gaussians.get_opacity.squeeze(-1)
    num = int(opacity.numel())
    if num == 0:
        return path
    sample_size = max(1, int(sample_frac * num))
    idx = torch.randperm(num, device=opacity.device)[:sample_size]

    def _safe_gather(tensor, fallback_nan=True):
        if tensor is None or tensor.numel() != num:
            if fallback_nan:
                return torch.full((sample_size,), float("nan"), device=opacity.device)
            return None
        return tensor[idx]

    importance = _safe_gather(gaussians.importance_score)
    visibility = _safe_gather(gaussians.visibility_score)
    error = _safe_gather(gaussians.error_score)
    raw_visibility = _safe_gather(raw_visibility)
    raw_importance = _safe_gather(raw_importance)

    opacity_cpu = opacity[idx].detach().cpu().numpy()
    importance_cpu = importance.detach().cpu().numpy()
    visibility_cpu = visibility.detach().cpu().numpy()
    error_cpu = error.detach().cpu().numpy()
    raw_visibility_cpu = raw_visibility.detach().cpu().numpy()
    raw_importance_cpu = raw_importance.detach().cpu().numpy()
    idx_cpu = idx.detach().cpu().numpy()

    expected_header = (
        "step,gaussian_idx,opacity,importance,visibility,error,raw_importance,raw_visibility\n"
    )
    header_needed = not os.path.exists(path)
    if not header_needed:
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline()
        if first_line and first_line != expected_header:
            base, ext = os.path.splitext(path)
            path = "{}_raw{}".format(base, ext or ".csv")
            header_needed = True
    with open(path, "a", encoding="utf-8") as f:
        if header_needed:
            f.write(expected_header)
        for i in range(sample_size):
            f.write(
                "{},{},{},{},{},{},{},{}\n".format(
                    int(iteration),
                    int(idx_cpu[i]),
                    float(opacity_cpu[i]),
                    float(importance_cpu[i]),
                    float(visibility_cpu[i]),
                    float(error_cpu[i]),
                    float(raw_importance_cpu[i]),
                    float(raw_visibility_cpu[i]),
                )
            )
    return path

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, full_args):
    if dataset.cap_max == -1:
        print("Please specify the maximum number of Gaussians using --cap_max.")
        exit()
    first_iter = 0
    use_wandb = WANDB_FOUND
    tb_writer = prepare_output_and_logger(full_args, use_wandb=use_wandb)
    gaussians = GaussianModel(dataset.sh_degree)
    noise_error_downscale = max(1, int(getattr(opt, "noise_error_downscale", 2)))
    resolution_scales = [1.0]
    if noise_error_downscale > 1:
        resolution_scales.append(float(noise_error_downscale))
    scene = Scene(dataset, gaussians, resolution_scales=resolution_scales)
    random_ply_path = getattr(scene.scene_info, "random_ply_path", None)
    gaussians.training_setup(opt)
    log_proxy_corr_all = getattr(args, "log_proxy_corr_all", False)
    correlation_analysis = bool(getattr(args, "correlation_analysis", False))
    corr_scatter_path = None
    corr_log_interval = 300
    print(
        "[Reloc] sampling={}, importance_ema={}, error_ema={}, importance_snapshot_top_frac={}, importance_update_interval={}, importance_subsample_stride={}, importance_subsample_ratio={}, log_proxy_corr_all={}, correlation_analysis={}".format(
            args.reloc_sampling,
            args.importance_ema,
            args.error_ema,
            args.importance_snapshot_top_frac,
            getattr(args, "importance_update_interval", 1),
            getattr(args, "importance_subsample_stride", 1),
            getattr(args, "importance_subsample_ratio", 1.0),
            log_proxy_corr_all,
            correlation_analysis,
        )
    )
    if tb_writer:
        tb_writer.add_text(
            "hparams/reloc_sampling",
            "sampling={}, importance_ema={}, error_ema={}, importance_snapshot_top_frac={}, importance_update_interval={}, importance_subsample_stride={}, importance_subsample_ratio={}, log_proxy_corr_all={}, correlation_analysis={}".format(
                args.reloc_sampling,
                args.importance_ema,
                args.error_ema,
                args.importance_snapshot_top_frac,
                getattr(args, "importance_update_interval", 1),
                getattr(args, "importance_subsample_stride", 1),
                getattr(args, "importance_subsample_ratio", 1.0),
                log_proxy_corr_all,
                correlation_analysis,
            ),
        )
    if use_wandb and wandb.run is not None:
        wandb.config.update(
            {
                "reloc_sampling": args.reloc_sampling,
                "importance_ema": args.importance_ema,
                "error_ema": args.error_ema,
                "importance_snapshot_top_frac": args.importance_snapshot_top_frac,
                "importance_update_interval": getattr(args, "importance_update_interval", 1),
                "importance_subsample_stride": getattr(args, "importance_subsample_stride", 1),
                "importance_subsample_ratio": getattr(args, "importance_subsample_ratio", 1.0),
                "log_proxy_corr_all": log_proxy_corr_all,
                "correlation_analysis": correlation_analysis,
            },
            allow_val_change=True,
        )
        scene_meta = {}
        if getattr(args, "scene_id", None):
            scene_meta["scene_id"] = str(args.scene_id)
        if getattr(args, "scene_type", None):
            scene_meta["scene_type"] = str(args.scene_type)
        if scene_meta:
            wandb.config.update(scene_meta, allow_val_change=True)
            _mlflow_log_params_safe(scene_meta)
        _mlflow_log_params_safe(
            {
                "reloc_sampling": args.reloc_sampling,
                "importance_ema": args.importance_ema,
                "error_ema": args.error_ema,
                "importance_snapshot_top_frac": args.importance_snapshot_top_frac,
                "importance_update_interval": getattr(args, "importance_update_interval", 1),
                "importance_subsample_stride": getattr(args, "importance_subsample_stride", 1),
                "importance_subsample_ratio": getattr(args, "importance_subsample_ratio", 1.0),
                "log_proxy_corr_all": log_proxy_corr_all,
                "correlation_analysis": correlation_analysis,
            }
        )
        _mlflow_log_params_safe(
            {
                "reloc_sampling": args.reloc_sampling,
                "importance_ema": args.importance_ema,
                "error_ema": args.error_ema,
                "importance_snapshot_top_frac": args.importance_snapshot_top_frac,
                "importance_update_interval": getattr(args, "importance_update_interval", 1),
                "importance_subsample_stride": getattr(args, "importance_subsample_stride", 1),
                "importance_subsample_ratio": getattr(args, "importance_subsample_ratio", 1.0),
                "log_proxy_corr_all": log_proxy_corr_all,
                "correlation_analysis": correlation_analysis,
            }
        )
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    base_cam_list = scene.getTrainCameras()
    fake_cam_list = (
        scene.getTrainCameras(scale=float(noise_error_downscale))
        if noise_error_downscale > 1
        else base_cam_list
    )
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    proxy_log_interval = max(1, opt.densification_interval)
    proxy_scatter_interval = 10000
    importance_update_interval = max(1, int(getattr(args, "importance_update_interval", 1)))
    prev_photometric_loss = None
    pending_reloc_prev_loss = None
    pending_reloc_iter = None
    psnr_threshold = float(getattr(args, "psnr_threshold", -1.0))
    time_to_threshold_logged = False
    train_start_wall_s = time.time()
    noise_guidance = normalize_noise_guidance(
        _noise_option(opt, full_args, "noise_guidance", default="opacity")
    )
    noise_percentile_threshold = float(
        _noise_option(opt, full_args, "noise_percentile_threshold", default=0.0)
    )
    noise_absolute_threshold = float(
        _noise_option(
            opt,
            full_args,
            "noise_absolute_threshold",
            "noise_error_absolute_threshold",
            default=0.005,
        )
    )
    noise_error_absolute_threshold = float(
        _noise_option(
            opt,
            full_args,
            "noise_error_absolute_threshold",
            "noise_absolute_threshold",
            default=noise_absolute_threshold,
        )
    )
    noise_amplification = float(
        _noise_option(opt, full_args, "noise_amplification", default=1.0)
    )
    per_pixel_error_metric = str(
        _noise_option(
            opt,
            full_args,
            "per_pixel_error_metric",
            "per_piexl_error_metric",
            default="l1",
        )
    ).lower()
    per_pixel_patch_size = int(
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
    use_error_guidance = uses_error_map(noise_guidance)
    error_avg = build_error_averager(avg_mode, window_size=avg_window, ema_decay=avg_ema_decay) if use_error_guidance else None
    fake_color = None
    error_contribution = None

    if correlation_analysis:
        corr_scatter_path = os.path.join(args.model_path, "correlation_scatter.csv")
        with torch.no_grad():
            corr_scatter_path = _append_corr_scatter_rows(
                corr_scatter_path, 0, gaussians, sample_frac=0.01
            )
        if use_wandb and wandb.run is not None:
            opacity_vals = gaussians.get_opacity.squeeze(-1)
            corr_sources = {
                "opacity": opacity_vals,
                "importance": gaussians.importance_score,
                "visibility": gaussians.visibility_score,
            }
            corr_log = {}
            for a_name, a_vals in corr_sources.items():
                for b_name, b_vals in corr_sources.items():
                    if a_vals is None or b_vals is None:
                        corr_log["corr/{}_{}".format(a_name, b_name)] = float("nan")
                        continue
                    if a_vals.numel() != b_vals.numel() or a_vals.numel() == 0:
                        corr_log["corr/{}_{}".format(a_name, b_name)] = float("nan")
                        continue
                    corr_log["corr/{}_{}".format(a_name, b_name)] = _safe_corrcoef(a_vals, b_vals)
            wandb.log(corr_log, step=0)
            _mlflow_log_metrics(corr_log, step=0)

    if 0 in testing_iterations:
        with torch.no_grad():
            zero = torch.tensor(0.0, device="cuda")
            training_report(
                tb_writer,
                0,
                zero,
                zero,
                l1_loss,
                0.0,
                testing_iterations,
                scene,
                render,
                (pipe, background),
            )

    for iteration in range(first_iter, opt.iterations + 1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = list(range(len(base_cam_list)))
        cam_idx = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        viewpoint_cam = base_cam_list[cam_idx]
        fake_viewpoint_cam = fake_cam_list[cam_idx]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]
        visibility_filter = render_pkg.get("visibility_filter")
        if visibility_filter is not None and visibility_filter.numel() == gaussians.get_xyz.shape[0]:
            with torch.no_grad():
                gaussians.update_visibility(visibility_filter, ema=args.visibility_ema)
        compute_importance = (
            args.reloc_sampling in ("importance", "importance_ema_quantile", "hybrid", "vis_importance", "vis_hybrid")
            or log_proxy_corr_all
        )
        if correlation_analysis:
            # Only compute importance when we will log correlation_scatter.csv.
            compute_importance = compute_importance or (iteration % corr_log_interval == 0)
        if compute_importance:
            max_id = render_pkg.get("max_id")
            if max_id is not None and (iteration % importance_update_interval == 0):
                with torch.no_grad():
                    max_id_sub, scale = _subsample_max_id(
                        max_id,
                        stride=getattr(args, "importance_subsample_stride", 1),
                        ratio=getattr(args, "importance_subsample_ratio", 1.0),
                    )
                    valid = max_id_sub >= 0
                    if args.importance_mode == "count":
                        if valid.any():
                            counts = torch.bincount(
                                max_id_sub[valid].view(-1),
                                minlength=gaussians.get_xyz.shape[0]
                            ).float() * scale
                        else:
                            counts = torch.zeros((gaussians.get_xyz.shape[0],), device="cuda")
                        gaussians.update_importance(counts, ema=args.importance_ema)
                    elif args.importance_mode == "wsum":
                        raise AssertionError("importance_mode=wsum is not supported without max_weight from renderer")
                    else:
                        raise AssertionError("Unknown importance_mode: {}".format(args.importance_mode))
        if args.reloc_sampling == "importance_snapshot":
            if (iteration < opt.densify_until_iter
                    and iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0):
                max_id = render_pkg.get("max_id")
                if max_id is not None:
                    with torch.no_grad():
                        valid = max_id >= 0
                        if valid.any():
                            counts = torch.bincount(
                                max_id[valid].view(-1),
                                minlength=gaussians.get_xyz.shape[0]
                            ).float()
                        else:
                            counts = torch.zeros((gaussians.get_xyz.shape[0],), device="cuda")
                        gaussians.update_importance_snapshot(
                            counts,
                            top_frac=args.importance_snapshot_top_frac
                        )
                else:
                    gaussians.clear_importance_snapshot()
            else:
                gaussians.clear_importance_snapshot()

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        fake_loss = None
        if use_error_guidance:
            if fake_color is None or fake_color.shape[0] != gaussians.get_xyz.shape[0]:
                fake_color = torch.zeros_like(gaussians.get_xyz, requires_grad=True, device="cuda")
                error_contribution = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
                if error_avg is not None:
                    error_avg.reset()
            black_bg = torch.zeros((3), device="cuda")
            fake_render = render(fake_viewpoint_cam, gaussians, pipe, black_bg, override_color=fake_color)["render"]
            # Detach image so fake_loss doesn't keep the main render graph alive.
            loss_per_pixel = compute_per_pixel_error_map(
                image.detach(),
                gt_image,
                metric=per_pixel_error_metric,
                patch_size=per_pixel_patch_size,
            )
            if fake_render.shape[-2:] != loss_per_pixel.shape:
                loss_per_pixel = torch.nn.functional.interpolate(
                    loss_per_pixel.unsqueeze(0).unsqueeze(0),
                    size=fake_render.shape[-2:],
                    mode="area",
                ).squeeze(0).squeeze(0)
            fake_loss = torch.sum((fake_render * loss_per_pixel).view(-1))
        Ll1 = l1_loss(image, gt_image)
        ssim_val = ssim(image, gt_image)
        train_psnr = psnr(image, gt_image).mean()
        photometric_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

        regularization_loss_opacity = args.opacity_reg * torch.abs(gaussians.get_opacity).mean()
        regularization_loss_covariance = args.scale_reg * torch.abs(gaussians.get_scaling).mean()
        loss = photometric_loss + regularization_loss_opacity + regularization_loss_covariance
        if fake_loss is not None:
            loss = loss + fake_loss

        loss.backward()
        if use_error_guidance and fake_color is not None and fake_color.grad is not None:
            with torch.no_grad():
                current_error = fake_color.grad[:, 0:1]
                if error_avg is not None:
                    error_contribution = error_avg.update(current_error)
                else:
                    error_contribution = current_error
                fake_color.grad = None
        if args.reloc_sampling == "error" or log_proxy_corr_all or correlation_analysis:
            if gaussians._opacity.grad is not None:
                with torch.no_grad():
                    gaussians.update_error_importance(
                        gaussians._opacity.grad.detach().abs().squeeze(-1),
                        ema=args.error_ema
                    )

        iter_end.record()

        with torch.no_grad():
            if pending_reloc_iter is not None and iteration == pending_reloc_iter + 1:
                delta_photometric_loss = float(photometric_loss.item()) - float(pending_reloc_prev_loss)
                if use_wandb and wandb.run is not None:
                    wandb.log({
                        "reloc/delta_photometric_loss": delta_photometric_loss,
                    }, step=pending_reloc_iter)
                    _mlflow_log_metrics(
                        {"reloc/delta_photometric_loss": delta_photometric_loss},
                        step=pending_reloc_iter,
                    )
                if tb_writer:
                    tb_writer.add_scalar("reloc/delta_photometric_loss", delta_photometric_loss, pending_reloc_iter)
                pending_reloc_iter = None
                pending_reloc_prev_loss = None
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
                final_wall_s = float(time.time() - train_start_wall_s)
                num_gaussians_final = int(gaussians.get_xyz.shape[0])
                opacity_vals_final = gaussians.get_opacity.squeeze(-1)
                num_gaussians_alive_final = int((opacity_vals_final > 0.005).sum().item())
                if use_wandb and wandb.run is not None:
                    wandb.log({
                        "train/total_time_s": final_wall_s,
                        "pop/num_final": num_gaussians_final,
                        "pop/num_final_million": num_gaussians_final / 1e6,
                        "pop/num_final_alive": num_gaussians_alive_final,
                    }, step=iteration)
                    _mlflow_log_metrics(
                        {
                            "train/total_time_s": final_wall_s,
                            "pop/num_final": num_gaussians_final,
                            "pop/num_final_million": num_gaussians_final / 1e6,
                            "pop/num_final_alive": num_gaussians_alive_final,
                        },
                        step=iteration,
                    )
                    wandb.run.summary["train/total_time_s"] = final_wall_s
                    wandb.run.summary["pop/num_final"] = num_gaussians_final
                    wandb.run.summary["pop/num_final_million"] = num_gaussians_final / 1e6
                    wandb.run.summary["pop/num_final_alive"] = num_gaussians_alive_final
                if tb_writer:
                    tb_writer.add_scalar("train/total_time_s", final_wall_s, iteration)
                    tb_writer.add_scalar("pop/num_final", num_gaussians_final, iteration)
                    tb_writer.add_scalar("pop/num_final_million", num_gaussians_final / 1e6, iteration)
                    tb_writer.add_scalar("pop/num_final_alive", num_gaussians_alive_final, iteration)

            # Log and save
            eval_metrics = training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
            )
            if (not time_to_threshold_logged) and psnr_threshold > 0:
                eval_test = eval_metrics.get("test", {})
                psnr_test = eval_test.get("psnr")
                if psnr_test is not None and psnr_test >= psnr_threshold:
                    time_to_threshold_s = float(time.time() - train_start_wall_s)
                    if use_wandb and wandb.run is not None:
                        wandb.log({
                            "eval/time_to_threshold_iter": iteration,
                            "eval/time_to_threshold_s": time_to_threshold_s,
                            "eval/psnr_threshold": psnr_threshold,
                        }, step=iteration)
                        _mlflow_log_metrics(
                            {
                                "eval/time_to_threshold_iter": iteration,
                                "eval/time_to_threshold_s": time_to_threshold_s,
                                "eval/psnr_threshold": psnr_threshold,
                            },
                            step=iteration,
                        )
                        wandb.run.summary["eval/time_to_threshold_iter"] = int(iteration)
                        wandb.run.summary["eval/time_to_threshold_s"] = float(time_to_threshold_s)
                        wandb.run.summary["eval/psnr_threshold"] = float(psnr_threshold)
                    if tb_writer:
                        tb_writer.add_scalar("eval/time_to_threshold_iter", iteration, iteration)
                        tb_writer.add_scalar("eval/time_to_threshold_s", time_to_threshold_s, iteration)
                        tb_writer.add_scalar("eval/psnr_threshold", psnr_threshold, iteration)
                    time_to_threshold_logged = True
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Per-iteration wandb metrics (before densify to avoid size mismatch)
            if use_wandb and wandb.run is not None:
                # Gaussian population / budget stats (cheap, every iter)
                opacity_vals = gaussians.get_opacity.squeeze(-1)
                alive_mask = opacity_vals > 0.005
                num_gaussians_alive = int(alive_mask.sum().item())
                num_gaussians_total_current = int(gaussians.get_xyz.shape[0])
                num_gaussians_dead = num_gaussians_total_current - num_gaussians_alive
                num_gaussians_total_budget = int(getattr(args, "cap_max", -1))
                alive_ratio_percent = 100.0 * (num_gaussians_alive / max(1, num_gaussians_total_current))

                # Visibility / importance proxies from renderer
                is_used = render_pkg.get("is_used", None)
                visibility_filter = render_pkg.get("visibility_filter", None)
                num_max_contributor_gaussians = 0
                num_visible_gaussians = 0
                mean_blending_weight = 0.0
                if is_used is not None and is_used.numel() == opacity_vals.numel():
                    num_max_contributor_gaussians = int(is_used.sum().item())
                if visibility_filter is not None and visibility_filter.numel() == opacity_vals.numel():
                    num_visible_gaussians = int(visibility_filter.sum().item())
                    visible_opacity = opacity_vals[visibility_filter]
                    if visible_opacity.numel() > 0:
                        mean_blending_weight = float(visible_opacity.mean().item())

                proxy_vals = None
                proxy_name = None
                if args.reloc_sampling == "random":
                    proxy_name = "random"
                elif args.reloc_sampling == "visibility":
                    proxy_vals = gaussians.visibility_score
                    proxy_name = "visibility"
                elif args.reloc_sampling == "opacity":
                    proxy_vals = opacity_vals
                    proxy_name = "opacity"
                elif args.reloc_sampling == "importance":
                    proxy_vals = gaussians.importance_score
                    proxy_name = "importance"
                elif args.reloc_sampling == "importance_snapshot":
                    proxy_vals = gaussians.importance_snapshot_score
                    proxy_name = "importance_snapshot"
                elif args.reloc_sampling == "importance_ema_quantile":
                    proxy_vals = gaussians.get_importance_ema_quantile()
                    proxy_name = "importance_ema_quantile"
                elif args.reloc_sampling == "error":
                    proxy_vals = gaussians.error_score
                    proxy_name = "error"
                elif args.reloc_sampling == "hybrid":
                    proxy_vals = opacity_vals * gaussians.importance_score
                    proxy_name = "hybrid"

                proxy_stats = {}
                if proxy_vals is not None and proxy_vals.numel() == opacity_vals.numel():
                    proxy_stats = _proxy_stats(proxy_vals, opacity_vals)
                proxy_corr_all = {}
                if log_proxy_corr_all:
                    proxy_corr_all = _proxy_corrs(opacity_vals, _all_proxy_values(gaussians, opacity_vals))

                wandb_log = {
                    # Core optimization signal
                    "loss/total": float(loss.item()),
                    # Loss decomposition
                    "loss/photometric": float(photometric_loss.item()),
                    "loss/reg_opacity": float(regularization_loss_opacity.item()),
                    "loss/reg_covariance": float(regularization_loss_covariance.item()),
                    # Rendering quality signals (training camera)
                    "quality/train_ssim": float(ssim_val.item()),
                    "quality/train_psnr": float(train_psnr.item()),
                    "loss/train_l1": float(Ll1.item()),
                    # Efficiency / optimizer
                    "perf/iter_ms": float(iter_start.elapsed_time(iter_end)),
                    "optim/xyz_lr": float(xyz_lr),
                    "pop/num_alive": num_gaussians_alive,
                    "pop/num_dead": num_gaussians_dead,
                    "pop/alive_ratio_percent": float(alive_ratio_percent),
                    # Visibility & importance (limited by available renderer outputs)
                    "vis/mean_blending_weight": float(mean_blending_weight),
                    "vis/num_max_contributor": num_max_contributor_gaussians,
                    "vis/num_visible": num_visible_gaussians,
                    "util/visible_over_alive": float(num_visible_gaussians / max(1, num_gaussians_alive)),
                    # Relocation proxy stats
                    "proxy/name": proxy_name,
                    "proxy/mean": proxy_stats.get("mean"),
                    "proxy/median": proxy_stats.get("median"),
                    "proxy/p95": proxy_stats.get("p95"),
                    "proxy/zero_frac": proxy_stats.get("zero_frac"),
                    "proxy/entropy": proxy_stats.get("entropy"),
                    "proxy/top1pct_share": proxy_stats.get("top1pct_share"),
                    "proxy/corr_opacity": proxy_stats.get("corr_opacity"),
                }
                if proxy_corr_all:
                    wandb_log.update({
                        "proxy_corr/{}".format(name): value for name, value in proxy_corr_all.items()
                    })
                if correlation_analysis and (iteration % corr_log_interval == 0):
                    corr_sources = {
                        "opacity": opacity_vals,
                        "importance": gaussians.importance_score,
                        "visibility": gaussians.visibility_score,
                    }
                    for a_name, a_vals in corr_sources.items():
                        for b_name, b_vals in corr_sources.items():
                            if a_vals is None or b_vals is None:
                                wandb_log["corr/{}_{}".format(a_name, b_name)] = float("nan")
                                continue
                            if a_vals.numel() != b_vals.numel() or a_vals.numel() == 0:
                                wandb_log["corr/{}_{}".format(a_name, b_name)] = float("nan")
                                continue
                            wandb_log["corr/{}_{}".format(a_name, b_name)] = _safe_corrcoef(a_vals, b_vals)
                wandb.log(wandb_log, step=iteration)
                _mlflow_log_metrics(wandb_log, step=iteration)

            if correlation_analysis and (iteration % corr_log_interval == 0):
                with torch.no_grad():
                    raw_visibility = None
                    if visibility_filter is not None and visibility_filter.numel() == gaussians.get_xyz.shape[0]:
                        raw_visibility = visibility_filter.float()
                    raw_importance = None
                    max_id = render_pkg.get("max_id")
                    if max_id is not None:
                        max_id_sub, scale = _subsample_max_id(
                            max_id,
                            stride=getattr(args, "importance_subsample_stride", 1),
                            ratio=getattr(args, "importance_subsample_ratio", 1.0),
                        )
                        valid = max_id_sub >= 0
                        if valid.any():
                            raw_importance = torch.bincount(
                                max_id_sub[valid].view(-1),
                                minlength=gaussians.get_xyz.shape[0],
                            ).float() * scale
                        else:
                            raw_importance = torch.zeros(
                                (gaussians.get_xyz.shape[0],), device="cuda"
                            )
                    corr_scatter_path = _append_corr_scatter_rows(
                        corr_scatter_path,
                        iteration,
                        gaussians,
                        raw_visibility=raw_visibility,
                        raw_importance=raw_importance,
                        sample_frac=0.01,
                    )

            if proxy_vals is not None and iteration % proxy_log_interval == 0:
                    if proxy_name != "opacity" and (iteration - 1) % proxy_scatter_interval == 0:
                        sample_size = max(1, proxy_vals.numel() // 100)
                        idx = torch.randperm(proxy_vals.numel(), device=proxy_vals.device)[:sample_size]
                        opacity_cpu = opacity_vals.detach()[idx].cpu().numpy()
                        proxy_cpu_scatter = proxy_vals.detach()[idx].cpu().numpy()
                        table = wandb.Table(data=list(zip(opacity_cpu, proxy_cpu_scatter)),
                                            columns=["opacity", "proxy"])
                        wandb.log({
                            "proxy/opacity_scatter": wandb.plot.scatter(
                                table, "opacity", "proxy",
                                title="Opacity vs {}".format(proxy_name)
                            )
                        }, step=iteration - 1)

            if tb_writer:
                tb_writer.add_scalar("train_loss_patches/psnr", train_psnr.item(), iteration)
                proxy_vals = None
                proxy_name = None
                if args.reloc_sampling == "random":
                    proxy_name = "random"
                elif args.reloc_sampling == "visibility":
                    proxy_vals = gaussians.visibility_score
                    proxy_name = "visibility"
                elif args.reloc_sampling == "opacity":
                    proxy_vals = gaussians.get_opacity.squeeze(-1)
                    proxy_name = "opacity"
                elif args.reloc_sampling == "importance":
                    proxy_vals = gaussians.importance_score
                    proxy_name = "importance"
                elif args.reloc_sampling == "importance_snapshot":
                    proxy_vals = gaussians.importance_snapshot_score
                    proxy_name = "importance_snapshot"
                elif args.reloc_sampling == "importance_ema_quantile":
                    proxy_vals = gaussians.get_importance_ema_quantile()
                    proxy_name = "importance_ema_quantile"
                elif args.reloc_sampling == "error":
                    proxy_vals = gaussians.error_score
                    proxy_name = "error"
                elif args.reloc_sampling == "hybrid":
                    proxy_vals = gaussians.get_opacity.squeeze(-1) * gaussians.importance_score
                    proxy_name = "hybrid"
                if proxy_vals is not None:
                    opacity_vals = gaussians.get_opacity.squeeze(-1)
                    proxy_stats = _proxy_stats(proxy_vals, opacity_vals)
                    tb_writer.add_scalar("proxy/mean", proxy_stats.get("mean"), iteration)
                    tb_writer.add_scalar("proxy/median", proxy_stats.get("median"), iteration)
                    tb_writer.add_scalar("proxy/p95", proxy_stats.get("p95"), iteration)
                    tb_writer.add_scalar("proxy/zero_frac", proxy_stats.get("zero_frac"), iteration)
                    tb_writer.add_scalar("proxy/entropy", proxy_stats.get("entropy"), iteration)
                    tb_writer.add_scalar("proxy/top1pct_share", proxy_stats.get("top1pct_share"), iteration)
                    tb_writer.add_scalar("proxy/corr_opacity", proxy_stats.get("corr_opacity"), iteration)
                    if iteration % proxy_log_interval == 0:
                        tb_writer.add_histogram("proxy/{}".format(proxy_name), proxy_vals, iteration)
                if log_proxy_corr_all:
                    opacity_vals = gaussians.get_opacity.squeeze(-1)
                    proxy_corr_all = _proxy_corrs(opacity_vals, _all_proxy_values(gaussians, opacity_vals))
                    for name, value in proxy_corr_all.items():
                        tb_writer.add_scalar("proxy_corr/{}".format(name), value, iteration)

            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
                reloc_info = gaussians.relocate_gs(dead_mask=dead_mask)
                add_info = gaussians.add_new_gs(cap_max=args.cap_max)
                if args.reloc_sampling == "importance_snapshot":
                    gaussians.clear_importance_snapshot()
                if prev_photometric_loss is not None:
                    pending_reloc_prev_loss = float(prev_photometric_loss)
                    pending_reloc_iter = iteration
                if use_wandb and wandb.run is not None:
                    wandb.log({
                        "reloc/num_dead": int(dead_mask.sum().item()),
                        "reloc/num_relocated": reloc_info.get("num_relocated"),
                        "reloc/mean_target_prob": reloc_info.get("mean_target_prob"),
                        "reloc/num_added": add_info.get("num_added"),
                        "reloc/mean_source_prob": add_info.get("mean_source_prob"),
                    }, step=iteration)
                    _mlflow_log_metrics(
                        {
                            "reloc/num_dead": int(dead_mask.sum().item()),
                            "reloc/num_relocated": reloc_info.get("num_relocated"),
                            "reloc/mean_target_prob": reloc_info.get("mean_target_prob"),
                            "reloc/num_added": add_info.get("num_added"),
                            "reloc/mean_source_prob": add_info.get("mean_source_prob"),
                        },
                        step=iteration,
                    )
                if tb_writer:
                    tb_writer.add_scalar("reloc/num_dead", int(dead_mask.sum().item()), iteration)
                    tb_writer.add_scalar("reloc/num_relocated", reloc_info.get("num_relocated"), iteration)
                    tb_writer.add_scalar("reloc/mean_target_prob", reloc_info.get("mean_target_prob"), iteration)
                    tb_writer.add_scalar("reloc/num_added", add_info.get("num_added"), iteration)
                    tb_writer.add_scalar("reloc/mean_source_prob", add_info.get("mean_source_prob"), iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if iteration < opt.iterations:
                L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                actual_covariance = L @ L.transpose(1, 2)
                if error_contribution is None or error_contribution.shape[0] != gaussians.get_xyz.shape[0]:
                    error_contribution = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
                noise_scale, noise_threshold = compute_noise_scale(
                    noise_guidance=noise_guidance,
                    opacity=gaussians.get_opacity,
                    importance_score=gaussians.importance_score,
                    error_contribution=error_contribution,
                    noise_percentile_threshold=noise_percentile_threshold,
                    noise_absolute_threshold=noise_absolute_threshold,
                    noise_error_absolute_threshold=noise_error_absolute_threshold,
                    noise_amplification=noise_amplification,
                )
                if tb_writer and noise_threshold is not None:
                    tb_writer.add_scalar("noise/noise_threshold", noise_threshold.item(), iteration)

                noise = torch.randn_like(gaussians._xyz) * noise_scale * args.noise_lr * xyz_lr
                noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                gaussians._xyz.add_(noise)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            prev_photometric_loss = float(photometric_loss.item())

    if getattr(full_args, "cleanup_random_ply", False):
        try:
            if random_ply_path and os.path.exists(random_ply_path):
                os.remove(random_ply_path)
        except Exception:
            pass
    if _MLFLOW_ACTIVE:
        mlflow.end_run()

def prepare_output_and_logger(args, use_wandb=True):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        if bool(getattr(args, "correlation_analysis", False)):
            reloc_tag = getattr(args, "reloc_sampling", "opacity")
            run_id = "cor_ana_{}_{}".format(reloc_tag, unique_str[:8])
        else:
            run_id = unique_str[0:10]
        args.model_path = os.path.join("./output/", run_id)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    mlflow_enabled = bool(getattr(args, "mlflow", False)) or use_wandb
    if use_wandb:
        if WANDB_FOUND:
            if getattr(args, "config", None):
                dataset_name = os.path.splitext(os.path.basename(args.config))[0]
            else:
                dataset_name = os.path.basename(os.path.normpath(args.source_path))
            init_type = getattr(args, "init_type", "random")
            project_name = "3dgs-mcmc-boosted-{}-{}".format(dataset_name, init_type)
            if bool(getattr(args, "correlation_analysis", False)):
                project_name = "{}-correlation-analysis".format(project_name)
            project_name = getattr(args, "wandb_project", None) or project_name
            wandb.init(
                project=project_name,
                config=vars(args),
                dir=args.model_path,
                resume="allow"
            )
            _mlflow_start_run(args, project_name, mlflow_enabled)
            if wandb.run is not None:
                if getattr(args, "wandb_run_name", None):
                    wandb.run.name = str(args.wandb_run_name)
                if wandb.run.name:
                    _mlflow_set_tag_safe("wandb_run_name", wandb.run.name)
                if wandb.run.name:
                    reloc_name = getattr(args, "reloc_sampling", "opacity")
                    importance_ema = getattr(args, "importance_ema", None)
                    error_ema = getattr(args, "error_ema", None)
                    ema_tag = ""
                    if reloc_name == "importance" and importance_ema is not None:
                        ema_tag = "-ema{}".format(importance_ema)
                    elif reloc_name == "error" and error_ema is not None:
                        ema_tag = "-ema{}".format(error_ema)
                    elif reloc_name == "hybrid" and importance_ema is not None and error_ema is not None:
                        ema_tag = "-ema{}-{}".format(importance_ema, error_ema)
                    cap_max = int(getattr(args, "cap_max", -1))
                    wandb.run.name = "{}{}-cap{}-{}".format(reloc_name, ema_tag, cap_max, wandb.run.name)
                    print("W&B run name: {}".format(wandb.run.name))
                    _mlflow_set_tag_safe("wandb_run_name", wandb.run.name)
                reloc_name = getattr(args, "reloc_sampling", "opacity")
                cap_max = int(getattr(args, "cap_max", -1))
                tags = list(wandb.run.tags) if wandb.run.tags else []
                tags.extend(["proxy:{}".format(reloc_name), "cap:{}".format(cap_max)])
                wandb.run.tags = list(dict.fromkeys(tags))
                _mlflow_set_tag_safe("wandb_tags", ",".join(wandb.run.tags))
                if wandb.run.summary.get("train_start_wall_s") is None:
                    wandb.run.summary["train_start_wall_s"] = float(time.time())
                if wandb.run.summary.get("pop/num_total") is None:
                    wandb.run.summary["pop/num_total"] = int(getattr(args, "cap_max", -1))
            print("W&B logging enabled")
        else:
            print("W&B not installed: skipping W&B logging")
            if mlflow_enabled:
                print("MLflow not started because W&B is disabled")
    else:
        if mlflow_enabled:
            if getattr(args, "config", None):
                dataset_name = os.path.splitext(os.path.basename(args.config))[0]
            else:
                dataset_name = os.path.basename(os.path.normpath(args.source_path))
            init_type = getattr(args, "init_type", "random")
            project_name = "3dgs-mcmc-boosted-{}-{}".format(dataset_name, init_type)
            if bool(getattr(args, "correlation_analysis", False)):
                project_name = "{}-correlation-analysis".format(project_name)
            _mlflow_start_run(args, project_name, mlflow_enabled)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    eval_metrics = {}
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                is_test = config['name'] == "test"
                lpips_model = None
                if is_test and LPIPS_FOUND:
                    lpips_model = _get_lpips_model(torch.device("cuda"))
                wandb_images = []
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    if WANDB_FOUND and wandb.run is not None and idx < 2:
                        img = (image.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        gt = (gt_image.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        wandb_images.append(wandb.Image(img, caption="{} render {}".format(config['name'], viewpoint.image_name)))
                        wandb_images.append(wandb.Image(gt, caption="{} gt {}".format(config['name'], viewpoint.image_name)))
                        _mlflow_log_image(
                            "eval/{}/render_{}".format(config['name'], viewpoint.image_name),
                            img,
                            iteration,
                        )
                        _mlflow_log_image(
                            "eval/{}/gt_{}".format(config['name'], viewpoint.image_name),
                            gt,
                            iteration,
                        )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    if is_test:
                        ssim_test += ssim(image, gt_image).mean().double()
                        if lpips_model is not None:
                            lpips_test += lpips_model(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                if is_test:
                    ssim_test /= len(config['cameras'])
                    if lpips_model is not None:
                        lpips_test /= len(config['cameras'])
                if is_test and lpips_model is not None:
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(
                        iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                elif is_test:
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {}".format(
                        iteration, config['name'], l1_test, psnr_test, ssim_test))
                else:
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    if is_test:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                        if lpips_model is not None:
                            tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                if WANDB_FOUND and wandb.run is not None:
                    wandb.run.summary["eval/{}/l1".format(config['name'])] = float(l1_test)
                    wandb.run.summary["eval/{}/psnr".format(config['name'])] = float(psnr_test)
                    if is_test:
                        wandb.run.summary["eval/{}/ssim".format(config['name'])] = float(ssim_test)
                        if lpips_model is not None:
                            wandb.run.summary["eval/{}/lpips".format(config['name'])] = float(lpips_test)
                    _mlflow_log_metrics(
                        {
                            "eval/{}/l1".format(config['name']): float(l1_test),
                            "eval/{}/psnr".format(config['name']): float(psnr_test),
                            "eval/{}/ssim".format(config['name']): float(ssim_test) if is_test else None,
                            "eval/{}/lpips".format(config['name']): float(lpips_test) if is_test and lpips_model is not None else None,
                        },
                        step=iteration,
                    )
                    if wandb_images:
                        wandb.log({
                            "eval/{}/images".format(config['name']): wandb_images
                        }, step=iteration)
                    wandb.log({
                        "eval/{}/l1".format(config['name']): float(l1_test),
                        "eval/{}/psnr".format(config['name']): float(psnr_test),
                    }, step=iteration)
                    _mlflow_log_metrics(
                        {
                            "eval/{}/l1".format(config['name']): float(l1_test),
                            "eval/{}/psnr".format(config['name']): float(psnr_test),
                        },
                        step=iteration,
                    )
                    if is_test:
                        wandb.log({
                            "eval/{}/ssim".format(config['name']): float(ssim_test),
                        }, step=iteration)
                        _mlflow_log_metrics(
                            {"eval/{}/ssim".format(config['name']): float(ssim_test)},
                            step=iteration,
                        )
                        if lpips_model is not None:
                            wandb.log({
                                "eval/{}/lpips".format(config['name']): float(lpips_test),
                            }, step=iteration)
                            _mlflow_log_metrics(
                                {"eval/{}/lpips".format(config['name']): float(lpips_test)},
                                step=iteration,
                            )
                    best_psnr_key = "eval/{}/psnr_best".format(config['name'])
                    best_psnr_iter_key = "eval/{}/psnr_best_iter".format(config['name'])
                    best_l1_key = "eval/{}/l1_best".format(config['name'])
                    best_l1_iter_key = "eval/{}/l1_best_iter".format(config['name'])
                    prev_best_psnr = wandb.run.summary.get(best_psnr_key, None)
                    prev_best_l1 = wandb.run.summary.get(best_l1_key, None)
                    if prev_best_psnr is None or float(psnr_test) > float(prev_best_psnr):
                        wandb.run.summary[best_psnr_key] = float(psnr_test)
                        wandb.run.summary[best_psnr_iter_key] = int(iteration)
                        _mlflow_log_metrics(
                            {
                                best_psnr_key: float(psnr_test),
                                best_psnr_iter_key: int(iteration),
                            },
                            step=iteration,
                        )
                    if prev_best_l1 is None or float(l1_test) < float(prev_best_l1):
                        wandb.run.summary[best_l1_key] = float(l1_test)
                        wandb.run.summary[best_l1_iter_key] = int(iteration)
                        _mlflow_log_metrics(
                            {
                                best_l1_key: float(l1_test),
                                best_l1_iter_key: int(iteration),
                            },
                            step=iteration,
                        )
                eval_metrics[config["name"]] = {
                    "l1": float(l1_test),
                    "psnr": float(psnr_test),
                }
                if is_test:
                    eval_metrics[config["name"]]["ssim"] = float(ssim_test)
                    if lpips_model is not None:
                        eval_metrics[config["name"]]["lpips"] = float(lpips_test)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
    return eval_metrics

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def _get_cli_keys(argv):
    keys = set()
    for token in argv:
        if token.startswith("--"):
            key = token[2:].split("=")[0]
            if key:
                keys.add(key)
        elif token.startswith("-") and len(token) > 1:
            key = token[1:2]
            if key:
                keys.add(key)
    return keys

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--scene_id", type=str, default=None)
    parser.add_argument("--scene_type", type=str, default=None)
    parser.add_argument("--mlflow", action="store_true", default=False)
    # Noise steering compatibility options from maxi version.
    parser.add_argument("--noise_amplification", type=float, default=None)
    parser.add_argument("--noise_percentile_threshold", type=float, default=None)
    parser.add_argument("--noise_absolute_threshold", type=float, default=None)
    parser.add_argument("--error_averaging", type=str, default=None)
    parser.add_argument("--moving_average_window_size", type=int, default=None)
    parser.add_argument("--noise_ema", type=float, default=None)
    parser.add_argument("--per_pixel_error_metric", type=str, default=None)
    parser.add_argument("--per_piexl_error_metric", type=str, default=None)
    parser.add_argument("--per_pixel_patch_size", type=int, default=None)
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    
    config = {}
    if args.config is not None:
        # Load the configuration file
        config = load_config(args.config)
        # Set config values only when not provided via CLI.
        cli_keys = _get_cli_keys(argv)
        for key, value in config.items():
            if key not in cli_keys:
                setattr(args, key, value)
    else:
        cli_keys = _get_cli_keys(argv)

    # Default test image logging: every 5k steps, including 0, unless explicitly set.
    if "test_iterations" not in cli_keys and "test_iterations" not in config:
        args.test_iterations = list(range(0, int(args.iterations) + 1, 5_000))
        if not args.test_iterations or args.test_iterations[-1] != int(args.iterations):
            args.test_iterations.append(int(args.iterations))

    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")
