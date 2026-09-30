from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class VideoDataset(Dataset):
    """Simple dataset for unimodal video clips and labels.

    Expects:
    - data: (N, C, T, H, W)
    - labels: (N, T)
    """

    def __init__(self, data: torch.Tensor, labels: torch.Tensor) -> None:
        if len(data) != len(labels):
            raise ValueError(f"Data/label length mismatch: {len(data)} vs {len(labels)}")
        if data.ndim != 5:
            raise ValueError(f"Expected data shape (N, C, T, H, W), got {tuple(data.shape)}")
        if labels.ndim != 2:
            raise ValueError(f"Expected labels shape (N, T), got {tuple(labels.shape)}")

        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]


class AMPNetDataset(Dataset):
    """Dataset for multimodal AMPNet inputs."""

    def __init__(self, rgb_data: torch.Tensor, thermal_data: torch.Tensor, labels: torch.Tensor) -> None:
        if not (len(rgb_data) == len(thermal_data) == len(labels)):
            raise ValueError("RGB, thermal, and labels must have the same number of samples.")
        self.rgb_data = rgb_data
        self.thermal_data = thermal_data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        rgb_sample = self.rgb_data[idx].clone().detach().float()
        thermal_sample = self.thermal_data[idx].clone().detach().float()
        label = self.labels[idx].clone().detach().float()
        return rgb_sample, thermal_sample, label


def create_ampnet_dataloader(data: torch.Tensor, labels: torch.Tensor, batch_size: int = 8, shuffle: bool = True) -> DataLoader:
    """Split a 4-channel tensor into RGB (0:3) and thermal (3:4) streams."""
    if data.ndim != 5:
        raise ValueError(f"Expected multimodal data shape (N, C, T, H, W), got {tuple(data.shape)}")
    if data.shape[1] < 4:
        raise ValueError("AMPNet expects at least 4 channels: 3 RGB + 1 thermal.")

    rgb_data = data[:, :3, :, :, :]
    thermal_data = data[:, 3:4, :, :, :]
    dataset = AMPNetDataset(rgb_data, thermal_data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
