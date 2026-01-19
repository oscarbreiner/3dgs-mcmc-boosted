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

def _subsample(tensor, max_samples):
    if tensor.numel() <= max_samples:
        return tensor
    idx = torch.randperm(tensor.numel(), device=tensor.device)[:max_samples]
    return tensor[idx]

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if dataset.cap_max == -1:
        print("Please specify the maximum number of Gaussians using --cap_max.")
        exit()
    first_iter = 0
    use_wandb = WANDB_FOUND
    tb_writer = prepare_output_and_logger(dataset, use_wandb=use_wandb)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    print(
        "[Reloc] sampling={}, importance_ema={}, error_ema={}".format(
            args.reloc_sampling, args.importance_ema, args.error_ema
        )
    )
    if tb_writer:
        tb_writer.add_text(
            "hparams/reloc_sampling",
            "sampling={}, importance_ema={}, error_ema={}".format(
                args.reloc_sampling, args.importance_ema, args.error_ema
            ),
        )
    if use_wandb and wandb.run is not None:
        wandb.config.update(
            {
                "reloc_sampling": args.reloc_sampling,
                "importance_ema": args.importance_ema,
                "error_ema": args.error_ema,
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
        if args.reloc_sampling in ("importance", "hybrid"):
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
                    gaussians.update_importance(counts, ema=args.importance_ema)

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_val = ssim(image, gt_image)
        photometric_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

        regularization_loss_opacity = args.opacity_reg * torch.abs(gaussians.get_opacity).mean()
        regularization_loss_covariance = args.scale_reg * torch.abs(gaussians.get_scaling).mean()
        loss = photometric_loss + regularization_loss_opacity + regularization_loss_covariance

        loss.backward()
        if args.reloc_sampling == "error":
            if gaussians._opacity.grad is not None:
                with torch.no_grad():
                    gaussians.update_error_importance(
                        gaussians._opacity.grad.detach().abs().squeeze(-1),
                        ema=args.error_ema
                    )

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
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
                if args.reloc_sampling == "opacity":
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

                wandb.log({
                    # Core optimization signal
                    "total_loss": float(loss.item()),
                    # Loss decomposition
                    "photometric_loss": float(photometric_loss.item()),
                    "regularization_loss_opacity": float(regularization_loss_opacity.item()),
                    "regularization_loss_covariance": float(regularization_loss_covariance.item()),
                    # Rendering quality signals (training camera)
                    "SSIM": float(ssim_val.item()),
                    # Efficiency / optimizer
                    "time/iter_ms": float(iter_start.elapsed_time(iter_end)),
                    "optim/xyz_lr": float(xyz_lr),
                    # Gaussian population
                    "num_gaussians_total": num_gaussians_total_budget,
                    "num_gaussians_alive": num_gaussians_alive,
                    "num_gaussians_dead": num_gaussians_dead,
                    "alive_ratio_percent": float(alive_ratio_percent),
                    # Visibility & importance (limited by available renderer outputs)
                    "mean_blending_weight": float(mean_blending_weight),
                    "num_max_contributor_gaussians": num_max_contributor_gaussians,
                    "num_visible_gaussians": num_visible_gaussians,
                    # Relocation proxy stats
                    "proxy/name": proxy_name,
                    "proxy/mean": proxy_stats.get("mean"),
                    "proxy/median": proxy_stats.get("median"),
                    "proxy/p95": proxy_stats.get("p95"),
                    "proxy/zero_frac": proxy_stats.get("zero_frac"),
                    "proxy/entropy": proxy_stats.get("entropy"),
                    "proxy/top1pct_share": proxy_stats.get("top1pct_share"),
                    "proxy/corr_opacity": proxy_stats.get("corr_opacity"),
                }, step=iteration)

                if proxy_vals is not None and iteration % proxy_log_interval == 0:
                    proxy_cpu = _subsample(proxy_vals.detach(), 20000).cpu().numpy()
                    wandb.log({
                        "proxy/hist": wandb.Histogram(proxy_cpu)
                    }, step=iteration)
                    if proxy_name != "opacity":
                        sample_size = min(5000, proxy_vals.numel())
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
                        }, step=iteration)

            if tb_writer:
                proxy_vals = None
                proxy_name = None
                if args.reloc_sampling == "opacity":
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

            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
                reloc_info = gaussians.relocate_gs(dead_mask=dead_mask)
                add_info = gaussians.add_new_gs(cap_max=args.cap_max)
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
            wandb.init(
                project="3dgs-mcmc-boosted",
                config=vars(args),
                dir=args.model_path,
                resume="allow"
            )
            if wandb.run is not None:
                if wandb.run.summary.get("train_start_wall_s") is None:
                    wandb.run.summary["train_start_wall_s"] = float(time.time())
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
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

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
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
