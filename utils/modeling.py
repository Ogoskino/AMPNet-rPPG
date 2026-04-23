from __future__ import annotations

import torch

from src.AMPNET import AMPNet, load_models
from src.EDSAN import EDSAN, R_3EDSAN, T_3EDSAN
from src.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from src.RTrPPG import N3DED64
from src.iBVPNet import iBVPNet



def build_model(model_name: str, device: torch.device):
    if model_name in ['AMPNet', 'AMPNet-NP']:
        rgb_models, thermal_models = load_models(R_3EDSAN, T_3EDSAN, device)
        return AMPNet(rgb_models, thermal_models, normalization=True).to(device)
    if model_name in ['T3EDSAN-NP-adLR', 'T3EDSAN-NP', 'T3EDSAN-CS', 'T3EDSAN-MSE']:
        return EDSAN(n_channels=1, model='thermal').to(device)
    if model_name in ['R3EDSAN-NP-adLR', 'R3EDSAN-NP', 'R3EDSAN', 'R3EDSAN-CS', 'R3EDSAN-MSE']:
        return EDSAN().to(device)
    if model_name == 'T3EDSAN-CBAM':
        return EDSAN(n_channels=1, model='thermal', is_cbam=True, is_tam=False).to(device)
    if model_name == 'T3EDSAN-TAM':
        return EDSAN(n_channels=1, model='thermal', is_cbam=False, is_tam=True).to(device)
    if model_name == 'T3ED':
        return EDSAN(n_channels=1, model='thermal', is_cbam=False, is_tam=False).to(device)
    if model_name == 'R3EDSAN-CBAM':
        return EDSAN(is_cbam=True, is_tam=False).to(device)
    if model_name == 'R3EDSAN-TAM':
        return EDSAN(is_cbam=False, is_tam=True).to(device)
    if model_name == 'R3ED':
        return EDSAN(is_cbam=False, is_tam=False).to(device)
    if model_name == 'iBVPNet':
        return iBVPNet(in_channels=3, frames=128, debug=True).to(device)
    if model_name == 'TiBVPNet':
        return iBVPNet(in_channels=1, frames=128, debug=True).to(device)
    if model_name == 'PhysNet':
        return PhysNet_padding_Encoder_Decoder_MAX().to(device)
    if model_name == 'TPhysNet':
        return PhysNet_padding_Encoder_Decoder_MAX(n_channels=1).to(device)
    if model_name == 'TRTrPPG-NP':
        return N3DED64(n_channels=1).to(device)
    return N3DED64().to(device)
