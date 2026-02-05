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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_loss_per_pixel(network_output, gt):
    """Compute L1 loss per pixel, returning a 2D map of per-pixel errors."""
    return torch.abs(network_output - gt).mean(dim=0)  # Average over RGB channels, keep spatial dims

def windowed_l1_per_pixel(network_output, gt, window_size=11):
    """
    Compute a windowed L1 loss per pixel by convolving a Gaussian window
    over the per-pixel L1 map to get the local average loss.
    Returns a tensor of the same spatial shape as the per-pixel L1 map.
    """
    per_pixel = torch.abs(network_output - gt)

    if per_pixel.dim() == 4:
        # (B, C, H, W) -> (B, 1, H, W)
        l1_map = per_pixel.mean(dim=1, keepdim=True)
    elif per_pixel.dim() == 3:
        # (C, H, W) -> (1, 1, H, W)
        l1_map = per_pixel.mean(dim=0, keepdim=True).unsqueeze(0)
    else:
        raise ValueError("Expected input with 3 or 4 dimensions (C,H,W) or (B,C,H,W).")

    window = create_window(window_size, channel=1)
    if l1_map.is_cuda:
        window = window.cuda(l1_map.get_device())
    window = window.type_as(l1_map)

    local_l1 = F.conv2d(l1_map, window, padding=window_size // 2, groups=1)

    if per_pixel.dim() == 3:
        return local_l1.squeeze(0).squeeze(0)
    return local_l1.squeeze(1)

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    

def psnr_per_pixel(img1, img2, window_size=11, max_val=1.0):
    """
    Computes a windowed (local) PSNR map.
    Returns a tensor of the same spatial shape as the input.
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    squared_error = (img1 - img2) ** 2
    local_mse = F.conv2d(squared_error, window, padding=window_size // 2, groups=channel)
    eps = 1e-10
    psnr_map = 10 * torch.log10((max_val ** 2) / (local_mse + eps))

    return psnr_map.mean(dim=-3)

