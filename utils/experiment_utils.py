from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset



def create_custom_folds(dataset_length: int, num_folds: int = 4):
    num_samples_per_group = dataset_length // num_folds
    indices = [np.arange(i * num_samples_per_group, (i + 1) * num_samples_per_group) for i in range(num_folds)]
    folds = []
    for i in range(num_folds):
        train_indices = np.hstack([indices[j] for j in range(num_folds) if j != i])
        test_indices = indices[i]
        folds.append((train_indices, test_indices))
    return folds



def extract_segments(data: torch.Tensor, ppg_signals: torch.Tensor, segment_length: int = 128):
    segments = []
    ppg_segments = []
    for i in range(data.shape[0]):
        for start_frame in range(0, data.shape[1] - segment_length + 1, segment_length):
            end_frame = start_frame + segment_length
            segments.append(data[i, start_frame:end_frame])
            ppg_segments.append(ppg_signals[i, start_frame:end_frame])
    return torch.stack(segments), torch.stack(ppg_segments)



def create_custom_dataloaders(dataset, batch_size: int, k: int = 4, mode: str = 'train'):
    folds = create_custom_folds(len(dataset), num_folds=k)
    dataloaders = []
    for i, (train_idx, val_idx) in enumerate(folds):
        if mode == 'train':
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            dataloaders.append(
                (
                    DataLoader(train_subset, batch_size=batch_size, shuffle=True),
                    DataLoader(val_subset, batch_size=batch_size, shuffle=False),
                )
            )
        elif i == 0:
            dataloaders.append((None, DataLoader(dataset, batch_size=batch_size, shuffle=False)))
    return dataloaders



def create_kfold_dataloaders(dataset, batch_size: int, k: int = 5, mode: str = 'train'):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    dataloaders = []
    for i, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        if mode == 'train':
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            dataloaders.append(
                (
                    DataLoader(train_subset, batch_size=batch_size, shuffle=True),
                    DataLoader(val_subset, batch_size=batch_size, shuffle=False),
                )
            )
        elif i == 0:
            dataloaders.append((None, DataLoader(dataset, batch_size=batch_size, shuffle=False)))
    return dataloaders
