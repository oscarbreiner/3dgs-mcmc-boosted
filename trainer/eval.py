import torch

from utils.image_utils import psnr
from utils.loss_utils import ssim
from trainer import logging as train_logging

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


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene,
    renderFunc,
    renderArgs,
):
    eval_metrics = {}
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                is_test = config["name"] == "test"
                lpips_model = None
                if is_test and LPIPS_FOUND:
                    lpips_model = _get_lpips_model(torch.device("cuda"))
                wandb_images = []
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0
                    )
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"] + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    if WANDB_FOUND and wandb.run is not None and idx < 2:
                        img = (image.detach().permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                        gt = (gt_image.detach().permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                        wandb_images.append(
                            wandb.Image(img, caption="{} render {}".format(config["name"], viewpoint.image_name))
                        )
                        wandb_images.append(
                            wandb.Image(gt, caption="{} gt {}".format(config["name"], viewpoint.image_name))
                        )
                        train_logging.mlflow_log_image(
                            "eval/{}/render_{}".format(config["name"], viewpoint.image_name),
                            img,
                            iteration,
                        )
                        train_logging.mlflow_log_image(
                            "eval/{}/gt_{}".format(config["name"], viewpoint.image_name),
                            gt,
                            iteration,
                        )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    if is_test:
                        ssim_test += ssim(image, gt_image).mean().double()
                        if lpips_model is not None:
                            lpips_test += lpips_model(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                if is_test:
                    ssim_test /= len(config["cameras"])
                    if lpips_model is not None:
                        lpips_test /= len(config["cameras"])
                if is_test and lpips_model is not None:
                    print(
                        "\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(
                            iteration, config["name"], l1_test, psnr_test, ssim_test, lpips_test
                        )
                    )
                elif is_test:
                    print(
                        "\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {}".format(
                            iteration, config["name"], l1_test, psnr_test, ssim_test
                        )
                    )
                else:
                    print(
                        "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                            iteration, config["name"], l1_test, psnr_test
                        )
                    )
                if tb_writer:
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)
                    if is_test:
                        tb_writer.add_scalar(config["name"] + "/loss_viewpoint - ssim", ssim_test, iteration)
                        if lpips_model is not None:
                            tb_writer.add_scalar(
                                config["name"] + "/loss_viewpoint - lpips", lpips_test, iteration
                            )
                if WANDB_FOUND and wandb.run is not None:
                    wandb.run.summary["eval/{}/l1".format(config["name"])] = float(l1_test)
                    wandb.run.summary["eval/{}/psnr".format(config["name"])] = float(psnr_test)
                    if is_test:
                        wandb.run.summary["eval/{}/ssim".format(config["name"])] = float(ssim_test)
                        if lpips_model is not None:
                            wandb.run.summary["eval/{}/lpips".format(config["name"])] = float(lpips_test)
                    if wandb_images:
                        wandb.log({"eval/{}/images".format(config["name"]): wandb_images}, step=iteration)
                    eval_log = {
                        "eval/{}/l1".format(config["name"]): float(l1_test),
                        "eval/{}/psnr".format(config["name"]): float(psnr_test),
                    }
                    if is_test:
                        eval_log["eval/{}/ssim".format(config["name"])] = float(ssim_test)
                    if is_test and lpips_model is not None:
                        eval_log["eval/{}/lpips".format(config["name"])] = float(lpips_test)
                    train_logging.log_metrics(
                        eval_log,
                        step=iteration,
                    )
                    best_psnr_key = "eval/{}/psnr_best".format(config["name"])
                    best_psnr_iter_key = "eval/{}/psnr_best_iter".format(config["name"])
                    best_l1_key = "eval/{}/l1_best".format(config["name"])
                    best_l1_iter_key = "eval/{}/l1_best_iter".format(config["name"])
                    prev_best_psnr = wandb.run.summary.get(best_psnr_key, None)
                    prev_best_l1 = wandb.run.summary.get(best_l1_key, None)
                    if prev_best_psnr is None or float(psnr_test) > float(prev_best_psnr):
                        wandb.run.summary[best_psnr_key] = float(psnr_test)
                        wandb.run.summary[best_psnr_iter_key] = int(iteration)
                        train_logging.mlflow_log_metrics(
                            {best_psnr_key: float(psnr_test), best_psnr_iter_key: int(iteration)},
                            step=iteration,
                        )
                    if prev_best_l1 is None or float(l1_test) < float(prev_best_l1):
                        wandb.run.summary[best_l1_key] = float(l1_test)
                        wandb.run.summary[best_l1_iter_key] = int(iteration)
                        train_logging.mlflow_log_metrics(
                            {best_l1_key: float(l1_test), best_l1_iter_key: int(iteration)},
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
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
    return eval_metrics
