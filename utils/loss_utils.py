from __future__ import annotations

import torch

SPECIAL_LOSS_MODELS = {
    'RTrPPG-NPSNR',
    'R3EDSAN-NP',
    'R3EDSAN-NP-adLR',
    'PhysNet',
    'RTrPPG-NP',
    'TRTrPPG-NPSNR',
    'T3EDSAN-NP',
    'T3EDSAN-NP-adLR',
    'TPhysNet',
    'TRTrPPG-NP',
    'RTrPPG',
}


def build_time_diff(inputs: torch.Tensor) -> torch.Tensor:
    seq_len = inputs.shape[2] if inputs.ndim == 5 else inputs.shape[1]
    return torch.linspace(0, 1, seq_len, device=inputs.device).unsqueeze(0).repeat(inputs.size(0), 1)



def compute_loss(model_name: str, criterion, outputs: torch.Tensor, labels: torch.Tensor, inputs: torch.Tensor):
    if model_name in SPECIAL_LOSS_MODELS:
        time_diff = build_time_diff(inputs)
        return criterion([outputs, labels, time_diff])
    return criterion(outputs, labels)
