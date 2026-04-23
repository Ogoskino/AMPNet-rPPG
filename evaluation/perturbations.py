import torch
import torch.nn.functional as F
from typing import Optional, List


def _check_video_tensor(x: torch.Tensor) -> None:
    """
    Ensure input is (N, C, T, H, W).
    """
    if x.ndim != 5:
        raise ValueError(f"Expected (N, C, T, H, W), got {tuple(x.shape)}")


# ---------------------------------------------------------
# 1) Resolution degradation
# ---------------------------------------------------------
def degrade_resolution(
    x: torch.Tensor,
    target_size: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Downsample each frame to target_size x target_size, then upsample back.

    Args:
        x: (N, C, T, H, W)
        target_size: e.g. 48, 32, 16
    """
    _check_video_tensor(x)

    n, c, t, h, w = x.shape

    frames = x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w)

    down = F.interpolate(
        frames,
        size=(target_size, target_size),
        mode=mode,
        align_corners=False if mode in ["bilinear", "bicubic"] else None,
    )

    up = F.interpolate(
        down,
        size=(h, w),
        mode=mode,
        align_corners=False if mode in ["bilinear", "bicubic"] else None,
    )

    out = up.reshape(n, t, c, h, w).permute(0, 2, 1, 3, 4)
    return out.contiguous()


# ---------------------------------------------------------
# 2) Gaussian noise
# ---------------------------------------------------------
def add_gaussian_noise(
    x: torch.Tensor,
    std: float = 0.03,
    clamp: bool = True,
) -> torch.Tensor:
    _check_video_tensor(x)

    noise = torch.randn_like(x) * std
    out = x + noise

    if clamp:
        out = torch.clamp(out, 0.0, 1.0)

    return out


# ---------------------------------------------------------
# 3) Motion blur (simple horizontal blur)
# ---------------------------------------------------------
def motion_blur(
    x: torch.Tensor,
    kernel_size: int = 5,
) -> torch.Tensor:
    """
    Apply simple motion blur across width.

    Note: lightweight approximation (fast + sufficient for experiments).
    """
    _check_video_tensor(x)

    if kernel_size <= 1:
        return x

    kernel = torch.zeros((kernel_size,))
    kernel[:] = 1.0 / kernel_size
    kernel = kernel.to(x.device)

    kernel = kernel.view(1, 1, 1, 1, kernel_size)

    # pad width
    padding = kernel_size // 2
    x_padded = F.pad(x, (padding, padding, 0, 0, 0, 0))

    out = F.conv3d(
        x_padded,
        kernel.expand(x.shape[1], 1, 1, 1, kernel_size),
        groups=x.shape[1],
    )

    return out


# ---------------------------------------------------------
# 4) Abrupt temporal resolution shifts
# ---------------------------------------------------------
def abrupt_resolution_shift(
    x: torch.Tensor,
    sizes: List[int],
    segment_length: int = 32,
) -> torch.Tensor:
    """
    Apply different resolution levels across time segments.

    Example:
        sizes = [64, 32, 16, 64]
    """
    _check_video_tensor(x)

    n, c, t, h, w = x.shape
    out = x.clone()

    for i, size in enumerate(sizes):
        start = i * segment_length
        end = min((i + 1) * segment_length, t)

        if start >= t:
            break

        if size == h:
            continue

        out[:, :, start:end] = degrade_resolution(
            out[:, :, start:end],
            target_size=size,
        )

    return out


# ---------------------------------------------------------
# 5) Wrapper (MAIN FUNCTION)
# ---------------------------------------------------------
def apply_perturbation(
    x: torch.Tensor,
    perturbation: Optional[str] = None,
    severity: str = "clean",
) -> torch.Tensor:
    """
    Apply controlled perturbation.

    Args:
        perturbation:
            None / "resolution" / "shift" / "gaussian" / "blur"
        severity:
            "clean", "mild", "moderate", "severe"
    """

    if perturbation is None or severity == "clean":
        return x

    # ---------------------------
    # Resolution degradation
    # ---------------------------
    if perturbation == "resolution":
        size_map = {
            "mild": 48,
            "moderate": 32,
            "severe": 16,
        }
        return degrade_resolution(x, target_size=size_map[severity])

    # ---------------------------
    # Gaussian noise
    # ---------------------------
    if perturbation == "gaussian":
        std_map = {
            "mild": 0.01,
            "moderate": 0.03,
            "severe": 0.05,
        }
        return add_gaussian_noise(x, std=std_map[severity])

    # ---------------------------
    # Motion blur
    # ---------------------------
    if perturbation == "blur":
        kernel_map = {
            "mild": 3,
            "moderate": 5,
            "severe": 7,
        }
        return motion_blur(x, kernel_size=kernel_map[severity])

    # ---------------------------
    # Temporal resolution shift
    # ---------------------------
    if perturbation == "shift":
        pattern_map = {
            "mild": [64, 48, 64, 48],
            "moderate": [64, 32, 64, 32],
            "severe": [64, 32, 16, 64],
        }
        return abrupt_resolution_shift(
            x,
            sizes=pattern_map[severity],
            segment_length=32,
        )

    raise ValueError(f"Unknown perturbation: {perturbation}")