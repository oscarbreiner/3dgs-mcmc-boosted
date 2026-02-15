import json
import math
import os
import time
import uuid

import numpy as np
import torch

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
    wandb = None

try:
    import mlflow

    MLFLOW_FOUND = True
except ImportError:
    MLFLOW_FOUND = False

_MLFLOW_ACTIVE = False
_MLFLOW_ENABLED = False


def mlflow_is_active():
    return _MLFLOW_ACTIVE


def mlflow_log_params_safe(params):
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


def mlflow_log_metrics(metrics, step=None):
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


def mlflow_set_tag_safe(key, value):
    if not _MLFLOW_ACTIVE:
        return
    if value is None:
        return
    mlflow.set_tag(str(key), str(value))


def mlflow_log_image(name, image_np, iteration):
    if not _MLFLOW_ACTIVE:
        return
    log_image = getattr(mlflow, "log_image", None)
    if log_image is None:
        return
    artifact_name = "{}/{}.png".format(int(iteration), name)
    log_image(image_np, artifact_file=artifact_name)


def mlflow_start_run(args, project_name, mlflow_enabled):
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
    mlflow_log_params_safe(vars(args))
    cfg_path = os.path.join(args.model_path, "cfg_args")
    if os.path.exists(cfg_path):
        mlflow.log_artifact(cfg_path)
    print("MLflow logging enabled")
    _MLFLOW_ACTIVE = True
    return True


def mlflow_end_run():
    global _MLFLOW_ACTIVE
    if _MLFLOW_ACTIVE:
        mlflow.end_run()
        _MLFLOW_ACTIVE = False


def wandb_is_active():
    return WANDB_FOUND and getattr(wandb, "run", None) is not None


def log_metrics(metrics, step=None, to_wandb=True):
    if to_wandb and wandb_is_active():
        wandb.log(metrics, step=step)
    mlflow_log_metrics(metrics, step=step)


def safe_corrcoef(x, y):
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


def proxy_stats(proxy_vals, opacity_vals):
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
    stats["corr_opacity"] = safe_corrcoef(proxy_vals, opacity_vals)
    return stats


def all_proxy_values(gaussians, opacity_vals):
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


def proxy_corrs(opacity_vals, proxy_map):
    corrs = {}
    for name, proxy_vals in proxy_map.items():
        if proxy_vals is None or proxy_vals.numel() != opacity_vals.numel():
            continue
        corrs[name] = safe_corrcoef(proxy_vals, opacity_vals)
    return corrs


def append_corr_scatter_rows(
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


def prepare_output_and_logger(args, use_wandb=True):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        if bool(getattr(args, "correlation_analysis", False)):
            reloc_tag = getattr(args, "reloc_sampling", "opacity")
            run_id = "cor_ana_{}_{}".format(reloc_tag, unique_str[:8])
        else:
            run_id = unique_str[0:10]
        args.model_path = os.path.join("./output/", run_id)

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(args))

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
                resume="allow",
            )
            mlflow_start_run(args, project_name, mlflow_enabled)
            if wandb.run is not None:
                if getattr(args, "wandb_run_name", None):
                    wandb.run.name = str(args.wandb_run_name)
                if wandb.run.name:
                    mlflow_set_tag_safe("wandb_run_name", wandb.run.name)
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
                    mlflow_set_tag_safe("wandb_run_name", wandb.run.name)
                reloc_name = getattr(args, "reloc_sampling", "opacity")
                cap_max = int(getattr(args, "cap_max", -1))
                tags = list(wandb.run.tags) if wandb.run.tags else []
                tags.extend(["proxy:{}".format(reloc_name), "cap:{}".format(cap_max)])
                wandb.run.tags = list(dict.fromkeys(tags))
                mlflow_set_tag_safe("wandb_tags", ",".join(wandb.run.tags))
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
            mlflow_start_run(args, project_name, mlflow_enabled)

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer
