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
import sys
import time
from argparse import ArgumentParser
from random import randint

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render, network_gui
from scene import GaussianModel, Scene
from trainer import logging as train_logging
from trainer.config import apply_config_overrides, apply_default_test_iterations
from trainer.eval import training_report
from trainer.noise import NoiseController, get_noise_error_downscale
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    import wandb
except ImportError:
    wandb = None

WANDB_FOUND = train_logging.WANDB_FOUND

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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, full_args):
    if dataset.cap_max == -1:
        print("Please specify the maximum number of Gaussians using --cap_max.")
        exit()
    args = full_args
    first_iter = 0
    use_wandb = WANDB_FOUND
    tb_writer = train_logging.prepare_output_and_logger(full_args, use_wandb=use_wandb)
    gaussians = GaussianModel(dataset.sh_degree)
    noise_error_downscale = get_noise_error_downscale(opt)
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
            train_logging.mlflow_log_params_safe(scene_meta)
        train_logging.mlflow_log_params_safe(
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
    logging_level = str(getattr(args, "logging_level", "core")).lower()
    if logging_level not in ("core", "diagnostic", "analysis"):
        logging_level = "core"
    diagnostic_logging = logging_level in ("diagnostic", "analysis")
    analysis_logging = logging_level == "analysis"
    noise_controller = NoiseController(opt, full_args)

    if correlation_analysis:
        corr_scatter_path = os.path.join(args.model_path, "correlation_scatter.csv")
        with torch.no_grad():
            corr_scatter_path = train_logging.append_corr_scatter_rows(
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
                    corr_log["corr/{}_{}".format(a_name, b_name)] = train_logging.safe_corrcoef(a_vals, b_vals)
            train_logging.log_metrics(corr_log, step=0)

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
        fake_loss = noise_controller.compute_fake_loss(
            image=image,
            gt_image=gt_image,
            fake_viewpoint_cam=fake_viewpoint_cam,
            gaussians=gaussians,
            pipe=pipe,
            render_fn=render,
        )
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
        noise_controller.update_after_backward()
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
                    train_logging.log_metrics(
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
                    train_logging.log_metrics(
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
                        train_logging.log_metrics(
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

            proxy_vals = None
            proxy_name = None

            # Per-iteration wandb metrics (before densify to avoid size mismatch)
            if use_wandb and wandb.run is not None:
                # Gaussian population / budget stats (cheap, every iter)
                opacity_vals = gaussians.get_opacity.squeeze(-1)
                alive_mask = opacity_vals > 0.005
                num_gaussians_alive = int(alive_mask.sum().item())
                num_gaussians_total_current = int(gaussians.get_xyz.shape[0])
                num_gaussians_dead = num_gaussians_total_current - num_gaussians_alive
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

                proxy_stats = {}
                proxy_corr_all = {}
                if diagnostic_logging:
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
                    if proxy_vals is not None and proxy_vals.numel() == opacity_vals.numel():
                        proxy_stats = train_logging.proxy_stats(proxy_vals, opacity_vals)
                if analysis_logging and log_proxy_corr_all:
                    proxy_corr_all = train_logging.proxy_corrs(
                        opacity_vals, train_logging.all_proxy_values(gaussians, opacity_vals)
                    )

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
                }
                if diagnostic_logging:
                    wandb_log.update(
                        {
                            "proxy/name": proxy_name,
                            "proxy/mean": proxy_stats.get("mean"),
                            "proxy/median": proxy_stats.get("median"),
                            "proxy/p95": proxy_stats.get("p95"),
                            "proxy/zero_frac": proxy_stats.get("zero_frac"),
                            "proxy/entropy": proxy_stats.get("entropy"),
                            "proxy/top1pct_share": proxy_stats.get("top1pct_share"),
                            "proxy/corr_opacity": proxy_stats.get("corr_opacity"),
                        }
                    )
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
                            wandb_log["corr/{}_{}".format(a_name, b_name)] = train_logging.safe_corrcoef(
                                a_vals, b_vals
                            )
                train_logging.log_metrics(wandb_log, step=iteration)

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
                    corr_scatter_path = train_logging.append_corr_scatter_rows(
                        corr_scatter_path,
                        iteration,
                        gaussians,
                        raw_visibility=raw_visibility,
                        raw_importance=raw_importance,
                        sample_frac=0.01,
                    )

            if analysis_logging and proxy_vals is not None and iteration % proxy_log_interval == 0:
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
                if diagnostic_logging and proxy_vals is not None:
                    opacity_vals = gaussians.get_opacity.squeeze(-1)
                    proxy_stats = train_logging.proxy_stats(proxy_vals, opacity_vals)
                    tb_writer.add_scalar("proxy/mean", proxy_stats.get("mean"), iteration)
                    tb_writer.add_scalar("proxy/median", proxy_stats.get("median"), iteration)
                    tb_writer.add_scalar("proxy/p95", proxy_stats.get("p95"), iteration)
                    tb_writer.add_scalar("proxy/zero_frac", proxy_stats.get("zero_frac"), iteration)
                    tb_writer.add_scalar("proxy/entropy", proxy_stats.get("entropy"), iteration)
                    tb_writer.add_scalar("proxy/top1pct_share", proxy_stats.get("top1pct_share"), iteration)
                    tb_writer.add_scalar("proxy/corr_opacity", proxy_stats.get("corr_opacity"), iteration)
                    if iteration % proxy_log_interval == 0:
                        tb_writer.add_histogram("proxy/{}".format(proxy_name), proxy_vals, iteration)
                if analysis_logging and log_proxy_corr_all:
                    opacity_vals = gaussians.get_opacity.squeeze(-1)
                    proxy_corr_all = train_logging.proxy_corrs(
                        opacity_vals, train_logging.all_proxy_values(gaussians, opacity_vals)
                    )
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
                    train_logging.log_metrics(
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
                noise_controller.apply_noise(
                    gaussians=gaussians,
                    args=args,
                    xyz_lr=xyz_lr,
                    iteration=iteration,
                    tb_writer=tb_writer,
                )

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
    train_logging.mlflow_end_run()

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
    parser.add_argument("--logging_level", type=str, default="core")
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
    
    args, config, cli_keys = apply_config_overrides(args, argv)
    apply_default_test_iterations(args, cli_keys, config)

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
