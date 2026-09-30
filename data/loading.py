from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

from utils.experiment_utils import extract_segments


def prepare_base_data(
    video_path: str | Path,
    label_path: str | Path,
    segment_length: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load saved tensors, segment them, and return videos as (N, C, T, H, W).

    Supports raw saved video tensors shaped:
    - (N, T, H, W, 3) RGB only
    - (N, T, H, W, 4) RGB + thermal
    """
    videos = torch.load(str(video_path), weights_only=True)
    labels = torch.load(str(label_path), weights_only=True)

    print("Raw videos shape:", videos.shape)
    print("Raw labels shape:", labels.shape)

    videos, labels = extract_segments(
        videos.float(),
        labels.float(),
        segment_length=segment_length,
    )

    # extract_segments returns (N, T, H, W, C)
    # Convert to (N, C, T, H, W)
    videos = videos.permute(0, 4, 1, 2, 3).contiguous()

    print("Segmented videos shape:", videos.shape)
    print("Segmented labels shape:", labels.shape)

    return videos, labels


def split_modalities(videos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Split video tensor into RGB and optional thermal.

    Supports:
    - RGB only:      (N, 3, T, H, W)
    - RGB + thermal: (N, 4, T, H, W)

    Returns:
    - rgb:     (N, 3, T, H, W)
    - thermal: (N, 1, T, H, W) or None
    """
    if videos.ndim != 5:
        raise ValueError(
            f"Expected shape (N, C, T, H, W), got {tuple(videos.shape)}"
        )

    channels = videos.shape[1]

    if channels == 3:
        return videos, None

    if channels >= 4:
        rgb = videos[:, 0:3, :, :, :]
        thermal = videos[:, 3:4, :, :, :]
        return rgb, thermal

    raise ValueError(f"Expected at least 3 channels for RGB, got {channels}")


def get_rgb_videos(videos: torch.Tensor) -> torch.Tensor:
    """Return RGB stream.

    Works for both:
    - RGB-only tensors with 3 channels
    - RGB+thermal tensors with 4 channels
    """
    rgb, _ = split_modalities(videos)
    return rgb


def get_thermal_videos(videos: torch.Tensor) -> torch.Tensor:
    """Return thermal stream.

    Raises a clear error if thermal channel is missing.
    """
    _, thermal = split_modalities(videos)

    if thermal is None:
        raise ValueError(
            "Thermal modality requested, but input has only RGB channels."
        )

    return thermal


def split_into_demographic_folds(
    videos: torch.Tensor,
    labels: torch.Tensor,
    num_groups: int,
):
    """Split contiguous groups for demographic-based evaluation."""
    if len(videos) != len(labels):
        raise ValueError(
            f"Video/label length mismatch: {len(videos)} vs {len(labels)}"
        )

    if num_groups <= 0:
        raise ValueError("num_groups must be greater than 0")

    fold_size = len(videos) // num_groups

    if fold_size == 0:
        raise ValueError(
            f"Cannot split {len(videos)} samples into {num_groups} groups."
        )

    video_folds = [
        videos[i * fold_size:(i + 1) * fold_size]
        for i in range(num_groups)
    ]

    label_folds = [
        labels[i * fold_size:(i + 1) * fold_size]
        for i in range(num_groups)
    ]

    return video_folds, label_folds
