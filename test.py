from __future__ import annotations

import mlflow
import numpy as np
import torch
import torch.nn as nn

from config import (
    BATCH_SIZE,
    DATASET_DIVISION,
    DEMOGRAPHIES,
    DEVICE,
    K_FOLDS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODALITY,
    NUM_DEMOGRAPHY_GROUPS,
    PATH_FOLDS,
    PERTURBATION_PLAN,
    RGB_MODEL_NAMES,
    SEGMENT_LENGTH,
    TEST_FEATURES_PATH,
    TEST_LABELS_PATH,
    THERMAL_MODEL_NAMES,
)
from data.dataset import VideoDataset, create_ampnet_dataloader 
from data.loading import get_rgb_videos, get_thermal_videos, prepare_base_data, split_into_demographic_folds
from evaluation.evaluate import compute_metrics, evaluate_ampnet_model, evaluate_model, log_and_visualize_results
from evaluation.loss import CosineSimilarityLoss, NPSNR
from evaluation.perturbations import apply_perturbation
from utils.experiment_utils import create_custom_dataloaders, create_kfold_dataloaders
from utils.modeling import build_model



def get_model_path(model_name: str, modality: str, fold: int) -> str:
    if modality == 'rgb':
        if model_name == 'R3EDSAN':
            return f'model_paths/best_model_{model_name}-MSE_fold_{fold+1}_ibvp_swise.pth'
        if model_name in ['R3EDSAN-TAM', 'R3ED']:
            return f'model_paths/best_model_{model_name}_fold_{fold+1}_ibvp_swise.pth'
        if model_name in ['iBVPNet', 'PhysNet']:
            return f'model_paths/best_model_{model_name}_fold_{fold+4}_ibvp_swise.pth'
        if model_name == 'RTrPPG':
            return f'model_paths/best_model_{model_name}-NP_fold_{fold+3}_ibvp_swise.pth'
        if model_name == 'R3EDSAN-NP':
            return f'model_paths/best_model_{model_name}_fold_{fold+3}_ibvp_swise.pth'
        if model_name == 'R3EDSAN-CS':
            return f'model_paths/best_model_{model_name}_fold_{fold+2}_ibvp_swise.pth'
        return f'model_paths/best_model_{model_name}_fold_{fold+2}_ibvp_swise.pth'
    if modality == 'thermal':
        if model_name in ['T3EDSAN-CS', 'T3EDSAN-NP', 'T3ED']:
            return f'model_paths/best_model_{model_name}_fold_{fold+4}_ibvp_thermal_swise.pth'
        if model_name == 'T3EDSAN-MSE':
            return f'model_paths/best_model_{model_name}_fold_{fold+3}_ibvp_thermal_swise.pth'
        if model_name == 'T3EDSAN-NP-adLR':
            return f'model_paths/best_model_{model_name}_fold_{fold+1}_ibvp_thermal_swise.pth'
        if model_name in ['TiBVPNet', 'TPhysNet', 'T3EDSAN-TAM', 'T3EDSAN-CBAM']:
            return f'model_paths/best_model_{model_name}_fold_{fold+2}_ibvp_thermal_swise.pth'
        return f'model_paths/best_model_{model_name}_fold_{fold+1}_ibvp_swise.pth'
    return f'model_paths/best_model_AMPNet_fold_{fold+1}.pth'



def get_criterion(model_name: str, criterion_cls):
    if model_name in ['RTrPPG-NPSNR', 'TRTrPPG-NPSNR']:
        return criterion_cls(Lambda=1.32)
    if model_name in ['R3EDSAN-NP-adLR', 'T3EDSAN-NP-adLR']:
        return criterion_cls(Lambda=0)
    if model_name in ['RTrPPG-NP', 'RTrPPG', 'R3EDSAN-NP', 'PhysNet', 'TRTrPPG-NP', 'T3EDSAN-NP', 'TPhysNet']:
        return criterion_cls(Lambda=0)
    return criterion_cls()



def test_model(model_path, dataloader_fn, dataset, batch_size, criterion, device, model_name, k, mode='train', demography=''):
    model = build_model(model_name, device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

    test_loss = 0.0
    all_outputs, all_labels, all_signals, all_metrics = [], [], [], []
    dataloaders = dataloader_fn(dataset, batch_size, k=k, mode=mode)

    with torch.no_grad():
        for _, val_dataloader in dataloaders:
            if model_name == 'AMPNet':
                val_loss, metrics, _, fold_outputs, _, fold_labels = evaluate_ampnet_model(model, val_dataloader, criterion, device, batch_size=batch_size)
                _, _, _, _, _, signal, _ = compute_metrics(fold_outputs, fold_labels)
            else:
                val_loss, metrics, fold_outputs, fold_labels = evaluate_model(model, val_dataloader, criterion, device, model_name)
                _, _, _, _, _, signal, _ = compute_metrics(np.asarray(fold_outputs), np.asarray(fold_labels))
            test_loss += val_loss * len(val_dataloader.dataset)
            all_outputs.append(fold_outputs)
            all_labels.append(fold_labels)
            all_metrics.append(metrics)
            all_signals.append(signal)

    test_loss /= len(dataset)
    torch.save(all_signals, f'{model_name}_outputs.pth')

    label = f'{model_name} {demography}'.strip()
    for i in range(len(all_outputs)):
        name = f'{model_name}_fold_{i}' if mode == 'train' else label
        log_and_visualize_results(all_outputs[i], all_metrics[i], all_labels[i], name)



def apply_multimodal_perturbation(videos, condition, perturbation=None, severity='clean'):
    out = videos.clone()
    if perturbation is None or severity == 'clean':
        return out
    if condition in {'rgb_only', 'both'}:
        out[:, 0:3] = apply_perturbation(out[:, 0:3], perturbation=perturbation, severity=severity)
    if condition in {'thermal_only', 'both'}:
        out[:, 3:4] = apply_perturbation(out[:, 3:4], perturbation=perturbation, severity=severity)
    if condition not in {'rgb_only', 'thermal_only', 'both'}:
        raise ValueError(f'Unknown condition: {condition}')
    return out



def run_unimodal_experiments(videos, labels, modality, model_names, criterion_classes, demography_name):
    dataloader_fn = create_kfold_dataloaders if DATASET_DIVISION == 'random' else create_custom_dataloaders
    for model_idx, model_name in enumerate(model_names):
        for fold in range(PATH_FOLDS):
            model_path = get_model_path(model_name, modality, fold)
            criterion = get_criterion(model_name, criterion_classes[model_idx])
            for perturbation, severities in PERTURBATION_PLAN.items():
                for severity in severities:
                    run_name = f'test_{model_name}_{demography_name}_{perturbation}_{severity}_fold_{fold+1}'
                    perturbed_videos = apply_perturbation(videos.clone(), perturbation=perturbation, severity=severity)
                    dataset = VideoDataset(data=perturbed_videos, labels=labels)
                    with mlflow.start_run(run_name=run_name):
                        mlflow.log_param('Model Name', model_name)
                        mlflow.log_param('Modality', modality)
                        mlflow.log_param('Demography', demography_name)
                        mlflow.log_param('Perturbation', perturbation if perturbation is not None else 'clean')
                        mlflow.log_param('Severity', severity)
                        mlflow.log_param('Fold', fold + 1)
                        test_model(model_path, dataloader_fn, dataset, BATCH_SIZE, criterion, DEVICE, model_name, K_FOLDS, mode='eval', demography=demography_name)



def run_multimodal_experiments(videos, labels, demography_name):
    dataloader_fn = create_kfold_dataloaders if DATASET_DIVISION == 'random' else create_custom_dataloaders
    model_name = 'AMPNet'
    criterion = nn.MSELoss()
    for fold in range(PATH_FOLDS):
        model_path = get_model_path(model_name, 'multimodal', fold)
        for condition in ['rgb_only', 'thermal_only', 'both']:
            for perturbation, severities in PERTURBATION_PLAN.items():
                for severity in severities:
                    run_name = f'test_{model_name}_{demography_name}_{condition}_{perturbation if perturbation is not None else "clean"}_{severity}_fold_{fold+1}'
                    perturbed_videos = apply_multimodal_perturbation(videos, condition=condition, perturbation=perturbation, severity=severity)
                    fusion_dataloader = create_ampnet_dataloader(perturbed_videos, labels, batch_size=BATCH_SIZE, shuffle=False)
                    dataset = fusion_dataloader.dataset
                    with mlflow.start_run(run_name=run_name):
                        mlflow.log_param('Model Name', model_name)
                        mlflow.log_param('Modality', 'multimodal')
                        mlflow.log_param('Demography', demography_name)
                        mlflow.log_param('Condition', condition)
                        mlflow.log_param('Perturbation', perturbation if perturbation is not None else 'clean')
                        mlflow.log_param('Severity', severity)
                        mlflow.log_param('Fold', fold + 1)
                        test_model(model_path, dataloader_fn, dataset, BATCH_SIZE, criterion, DEVICE, model_name, K_FOLDS, mode='eval', demography=demography_name)


if __name__ == '__main__':
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT_NAME)

    videos, labels = prepare_base_data(TEST_FEATURES_PATH, TEST_LABELS_PATH, segment_length=SEGMENT_LENGTH)

    if MODALITY == 'rgb':
        videos = get_rgb_videos(videos)
        video_folds, label_folds = split_into_demographic_folds(videos, labels, NUM_DEMOGRAPHY_GROUPS)
        criterion_classes = [nn.MSELoss, NPSNR, CosineSimilarityLoss, NPSNR]
        for idx, demography_name in enumerate(DEMOGRAPHIES[:NUM_DEMOGRAPHY_GROUPS]):
            run_unimodal_experiments(video_folds[idx], label_folds[idx], 'rgb', RGB_MODEL_NAMES, criterion_classes, demography_name)
    elif MODALITY == 'thermal':
        videos = get_thermal_videos(videos)
        video_folds, label_folds = split_into_demographic_folds(videos, labels, NUM_DEMOGRAPHY_GROUPS)
        criterion_classes = [CosineSimilarityLoss, CosineSimilarityLoss, CosineSimilarityLoss]
        for idx, demography_name in enumerate(DEMOGRAPHIES[:NUM_DEMOGRAPHY_GROUPS]):
            run_unimodal_experiments(video_folds[idx], label_folds[idx], 'thermal', THERMAL_MODEL_NAMES, criterion_classes, demography_name)
    elif MODALITY == 'multimodal':
        video_folds, label_folds = split_into_demographic_folds(videos, labels, NUM_DEMOGRAPHY_GROUPS)
        for idx, demography_name in enumerate(DEMOGRAPHIES[:NUM_DEMOGRAPHY_GROUPS]):
            run_multimodal_experiments(video_folds[idx], label_folds[idx], demography_name)
    else:
        raise ValueError(f'Unsupported modality: {MODALITY}')
