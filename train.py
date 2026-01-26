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
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
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
    from lpipsPyTorch import LPIPS
    LPIPS_FOUND = True
except ImportError:
    LPIPS_FOUND = False

_LPIPS_NET_TYPE = "vgg"
_LPIPS_MODEL = None

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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, full_args):
    if dataset.cap_max == -1:
        print("Please specify the maximum number of Gaussians using --cap_max.")
        exit()
    first_iter = 0
    use_wandb = WANDB_FOUND
    tb_writer = prepare_output_and_logger(full_args, use_wandb=use_wandb)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    log_proxy_corr_all = getattr(args, "log_proxy_corr_all", False)
    print(
        "[Reloc] sampling={}, importance_ema={}, error_ema={}, log_proxy_corr_all={}".format(
            args.reloc_sampling, args.importance_ema, args.error_ema, log_proxy_corr_all
        )
    )
    if tb_writer:
        tb_writer.add_text(
            "hparams/reloc_sampling",
            "sampling={}, importance_ema={}, error_ema={}, log_proxy_corr_all={}".format(
                args.reloc_sampling, args.importance_ema, args.error_ema, log_proxy_corr_all
            ),
        )
    if use_wandb and wandb.run is not None:
        wandb.config.update(
            {
                "reloc_sampling": args.reloc_sampling,
                "importance_ema": args.importance_ema,
                "error_ema": args.error_ema,
                "log_proxy_corr_all": log_proxy_corr_all,
            },
            allow_val_change=True,
        )
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    proxy_log_interval = max(1, opt.densification_interval)
    proxy_scatter_interval = 10000
    prev_photometric_loss = None
    pending_reloc_prev_loss = None
    pending_reloc_iter = None
    psnr_threshold = float(getattr(args, "psnr_threshold", -1.0))
    time_to_threshold_logged = False
    train_start_wall_s = time.time()

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
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

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
        if args.reloc_sampling in ("importance", "hybrid", "vis_importance", "vis_hybrid") or log_proxy_corr_all:
            max_id = render_pkg.get("max_id")
            if max_id is not None:
                with torch.no_grad():
                    valid = max_id >= 0
                    if args.importance_mode == "count":
                        if valid.any():
                            counts = torch.bincount(
                                max_id[valid].view(-1),
                                minlength=gaussians.get_xyz.shape[0]
                            ).float()
                        else:
                            counts = torch.zeros((gaussians.get_xyz.shape[0],), device="cuda")
                        gaussians.update_importance(counts, ema=args.importance_ema)
                    elif args.importance_mode == "wsum":
                        raise AssertionError("importance_mode=wsum is not supported without max_weight from renderer")
                    else:
                        raise AssertionError("Unknown importance_mode: {}".format(args.importance_mode))

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_val = ssim(image, gt_image)
        train_psnr = psnr(image, gt_image).mean()
        photometric_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

        regularization_loss_opacity = args.opacity_reg * torch.abs(gaussians.get_opacity).mean()
        regularization_loss_covariance = args.scale_reg * torch.abs(gaussians.get_scaling).mean()
        loss = photometric_loss + regularization_loss_opacity + regularization_loss_covariance

        loss.backward()
        if args.reloc_sampling == "error" or log_proxy_corr_all:
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
                wandb.log(wandb_log, step=iteration)

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

                def op_sigmoid(x, k=100, x0=0.995):
                    return 1 / (1 + torch.exp(-k * (x - x0)))

                noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity))*args.noise_lr*xyz_lr
                noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                gaussians._xyz.add_(noise)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            prev_photometric_loss = float(photometric_loss.item())

def prepare_output_and_logger(args, use_wandb=True):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if use_wandb:
        if WANDB_FOUND:
            if getattr(args, "config", None):
                dataset_name = os.path.splitext(os.path.basename(args.config))[0]
            else:
                dataset_name = os.path.basename(os.path.normpath(args.source_path))
            init_type = getattr(args, "init_type", "random")
            project_name = "3dgs-mcmc-boosted-{}-{}".format(dataset_name, init_type)
            wandb.init(
                project=project_name,
                config=vars(args),
                dir=args.model_path,
                resume="allow"
            )
            if wandb.run is not None:
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
                reloc_name = getattr(args, "reloc_sampling", "opacity")
                cap_max = int(getattr(args, "cap_max", -1))
                tags = list(wandb.run.tags) if wandb.run.tags else []
                tags.extend(["proxy:{}".format(reloc_name), "cap:{}".format(cap_max)])
                wandb.run.tags = list(dict.fromkeys(tags))
                if wandb.run.summary.get("train_start_wall_s") is None:
                    wandb.run.summary["train_start_wall_s"] = float(time.time())
                if wandb.run.summary.get("pop/num_total") is None:
                    wandb.run.summary["pop/num_total"] = int(getattr(args, "cap_max", -1))
            print("W&B logging enabled")
        else:
            print("W&B not installed: skipping W&B logging")

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
                    if wandb_images:
                        wandb.log({
                            "eval/{}/images".format(config['name']): wandb_images
                        }, step=iteration)
                    wandb.log({
                        "eval/{}/l1".format(config['name']): float(l1_test),
                        "eval/{}/psnr".format(config['name']): float(psnr_test),
                    }, step=iteration)
                    if is_test:
                        wandb.log({
                            "eval/{}/ssim".format(config['name']): float(ssim_test),
                        }, step=iteration)
                        if lpips_model is not None:
                            wandb.log({
                                "eval/{}/lpips".format(config['name']): float(lpips_test),
                            }, step=iteration)
                    best_psnr_key = "eval/{}/psnr_best".format(config['name'])
                    best_psnr_iter_key = "eval/{}/psnr_best_iter".format(config['name'])
                    best_l1_key = "eval/{}/l1_best".format(config['name'])
                    best_l1_iter_key = "eval/{}/l1_best_iter".format(config['name'])
                    prev_best_psnr = wandb.run.summary.get(best_psnr_key, None)
                    prev_best_l1 = wandb.run.summary.get(best_l1_key, None)
                    if prev_best_psnr is None or float(psnr_test) > float(prev_best_psnr):
                        wandb.run.summary[best_psnr_key] = float(psnr_test)
                        wandb.run.summary[best_psnr_iter_key] = int(iteration)
                    if prev_best_l1 is None or float(l1_test) < float(prev_best_l1):
                        wandb.run.summary[best_l1_key] = float(l1_test)
                        wandb.run.summary[best_l1_iter_key] = int(iteration)
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
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    
    if args.config is not None:
        # Load the configuration file
        config = load_config(args.config)
        # Set config values only when not provided via CLI.
        cli_keys = _get_cli_keys(argv)
        for key, value in config.items():
            if key not in cli_keys:
                setattr(args, key, value)

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
