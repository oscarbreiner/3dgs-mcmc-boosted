import torch
import torch.nn.functional as F


class WindowedAverage:
    def __init__(self, window_size):
        self.window_size = max(1, int(window_size))
        self._buffer = []
        self._sum = None
        self._shape = None

    def reset(self):
        self._buffer = []
        self._sum = None
        self._shape = None

    def update(self, x):
        if self._shape is None or x.shape != self._shape:
            self.reset()
            self._shape = x.shape
        if self._sum is None:
            self._sum = torch.zeros_like(x)
        x_detached = x.detach()
        self._buffer.append(x_detached)
        self._sum = self._sum + x_detached
        if len(self._buffer) > self.window_size:
            oldest = self._buffer.pop(0)
            self._sum = self._sum - oldest
        return self._sum / float(len(self._buffer))


class ExponentialMovingAverage:
    def __init__(self, decay):
        self.decay = float(decay)
        self._value = None
        self._shape = None

    def reset(self):
        self._value = None
        self._shape = None

    def update(self, x):
        if self._shape is None or x.shape != self._shape:
            self.reset()
            self._shape = x.shape
        x_detached = x.detach()
        if self._value is None:
            self._value = x_detached
        else:
            self._value = self.decay * self._value + (1.0 - self.decay) * x_detached
        return self._value


def sigmoid_guidance(x, k=100, x0=0.995):
    return 1 / (1 + torch.exp(-k * (x - x0)))


def _avg_pool_map(map_2d, patch_size):
    if patch_size <= 1:
        return map_2d
    return F.avg_pool2d(
        map_2d.unsqueeze(0).unsqueeze(0),
        kernel_size=patch_size,
        stride=1,
        padding=patch_size // 2,
    ).squeeze(0).squeeze(0)


def compute_per_pixel_error_map(image, gt_image, metric="l1", patch_size=1):
    metric = str(metric).lower()
    patch_size = max(1, int(patch_size))
    if metric == "psnr":
        mse = (image - gt_image).pow(2).mean(dim=0)
        mse = _avg_pool_map(mse, patch_size)
        return 10.0 * torch.log10(1.0 / (mse + 1e-10))
    l1_map = torch.abs(image - gt_image).mean(dim=0)
    return _avg_pool_map(l1_map, patch_size)


def uses_error_map(noise_guidance):
    guidance = str(noise_guidance or "opacity").lower()
    return guidance in (
        "error",
        "opacity_error_percentile",
        "opacity_error_threshold",
    )

def build_error_averager(avg_mode, window_size=100, ema_decay=0.9):
    mode = str(avg_mode or "windowed_moving_average").lower()
    if mode == "windowed_moving_average":
        return WindowedAverage(window_size)
    if mode == "ema":
        return ExponentialMovingAverage(ema_decay)
    if mode == "none":
        return None
    raise ValueError(f"Unknown noise_error_avg_mode '{avg_mode}'")


def compute_noise_scale(
    noise_guidance,
    opacity,
    error_contribution,
    noise_error_percentile_threshold=0.0,
    noise_error_absolute_threshold=0.005,
    noise_amplification=1.0,
):
    guidance = str(noise_guidance or "opacity").lower()
    amplification = float(noise_amplification)
    percentile_threshold = float(noise_error_percentile_threshold)
    absolute_threshold = float(noise_error_absolute_threshold)

    if guidance == "opacity":
        return sigmoid_guidance(1 - opacity, k=100, x0=0.995) * amplification, None
    if guidance == "error" and percentile_threshold > 0:
        threshold = torch.quantile(error_contribution.squeeze(-1), percentile_threshold)
        return sigmoid_guidance(error_contribution, 100, threshold) * amplification, threshold
    if guidance == "error":
        return sigmoid_guidance(error_contribution, 100, absolute_threshold) * amplification, None
    if guidance == "opacity_error_percentile":
        if percentile_threshold <= 0:
            raise ValueError(
                "noise_error_percentile_threshold must be > 0 when noise_guidance='opacity_error_percentile'"
            )
        threshold = torch.quantile(error_contribution.squeeze(-1), percentile_threshold)
        return (
            sigmoid_guidance(error_contribution, 100, threshold)
            * sigmoid_guidance(1 - opacity, k=100, x0=0.995)
            * amplification,
            threshold,
        )
    if guidance == "opacity_error_threshold":
        return (
            sigmoid_guidance(error_contribution, 100, absolute_threshold)
            * sigmoid_guidance(1 - opacity, k=100, x0=0.995)
            * amplification,
            None,
        )
    if guidance == "random":
        return 1.0 * amplification, None
    raise ValueError(f"Unknown noise guidance '{noise_guidance}' (normalized: '{guidance}')")
