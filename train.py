from __future__ import annotations

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mlflow.models.signature import infer_signature
from tabulate import tabulate

from config import (
    BATCH_SIZE,
    DEVICE,
    K_FOLDS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODALITY,
    SEGMENT_LENGTH,
    TRAIN_FEATURES_PATH,
    TRAIN_LABELS_PATH,
    TRAIN_RGB_MODEL_NAMES,
    TRAIN_THERMAL_MODEL_NAMES,
)
from data import VideoDataset, create_ampnet_dataloader, get_rgb_videos, get_thermal_videos, prepare_base_data
from evaluation.evaluate import evaluate_ampnet_model, evaluate_model, log_model_summary, plot_bvp, plot_hr_sp_bap, save_best_model
from evaluation.loss import CosineSimilarityLoss, NPSNR
from src.AMPNET import AMPNet, load_models
from src.EDSAN import EDSAN, R_3EDSAN, T_3EDSAN
from src.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from src.RTrPPG import N3DED64
from src.iBVPNet import iBVPNet
from utils.experiment_utils import create_custom_dataloaders
from utils.loss_utils import compute_loss
from utils.modeling import build_model


MODEL_CLASS_MAP = {
    'R3EDSAN-CBAM': EDSAN,
    'R3EDSAN-TAM': EDSAN,
    'R3ED': EDSAN,
    'T3EDSAN-CBAM': EDSAN,
    'T3EDSAN-TAM': EDSAN,
    'T3ED': EDSAN,
    'R3EDSAN': EDSAN,
    'PhysNet': PhysNet_padding_Encoder_Decoder_MAX,
    'iBVPNet': iBVPNet,
    'RTrPPG': N3DED64,
}



def get_optimizer_and_criterion(model_name: str, model, criterion_cls):
    if model_name in ['RTrPPG-NPSNR', 'TRTrPPG-NPSNR']:
        criterion = criterion_cls(Lambda=1.32)
        lr = 0.00044
    elif model_name in ['R3EDSAN-NP-adLR', 'T3EDSAN-NP-adLR']:
        criterion = criterion_cls(Lambda=0)
        lr = 0.00044
    elif model_name in ['RTrPPG-NP', 'R3EDSAN-NP', 'PhysNet', 'TRTrPPG-NP', 'T3EDSAN-NP', 'TPhysNet', 'RTrPPG']:
        criterion = criterion_cls(Lambda=0)
        lr = 0.0001
    else:
        criterion = criterion_cls()
        lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return criterion, optimizer, None



def log_trained_model(model, model_name, input_sample, output_sample):
    output_sample_np = output_sample.detach().cpu().numpy()
    input_sample_np = input_sample.cpu().numpy()
    signature = infer_signature(input_sample_np, output_sample_np)
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path=model_name,
        input_example=input_sample_np,
        signature=signature,
    )



def train_one_epoch(model, dataloader, criterion, optimizer, device, model_name: str):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = compute_loss(model_name, criterion, outputs, labels, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)



def train_one_ampnet_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for rgb_inputs, thermal_inputs, labels in dataloader:
        rgb_inputs = rgb_inputs.to(device)
        thermal_inputs = thermal_inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs, _, _ = model(rgb_inputs, thermal_inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * rgb_inputs.size(0)
    return running_loss / len(dataloader.dataset)



def train_unimodal(models, model_names, criterion_classes, data, labels, n_splits=1, batch_size=BATCH_SIZE, num_epochs=40, data_name='ibvp_swise'):
    fold_results_dict = {name: [] for name in model_names}
    average_results_dict = {name: {} for name in model_names}
    dataset = VideoDataset(data=data, labels=labels)

    for idx, model_name in enumerate(model_names):
        model_class = models[idx]
        with mlflow.start_run(run_name=model_name, nested=True):
            mlflow.log_param('Model Name', model_name)
            mlflow.log_param('K-Folds', n_splits)
            fold_results = []
            dataloaders = create_custom_dataloaders(dataset, batch_size=batch_size, k=n_splits)

            for fold, (train_dataloader, val_dataloader) in enumerate(dataloaders):
                model = build_model(model_name, DEVICE)
                criterion, optimizer, scheduler = get_optimizer_and_criterion(model_name, model, criterion_classes[idx])
                if fold == 0:
                    log_model_summary(model, model_name, (8, data.shape[1], data.shape[2], data.shape[3], data.shape[4]), DEVICE)

                best_val_rmse = float('inf')
                best_metrics = {}

                for epoch in range(num_epochs):
                    train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, DEVICE, model_name)
                    val_loss, metrics, all_outputs, all_labels = evaluate_model(model, val_dataloader, criterion, DEVICE, model_name)
                    mae, rmse, pcc, snr_pred, macc, _, _ = metrics
                    if scheduler:
                        scheduler.step()
                    print(f'Model {model_name}, Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
                    if rmse < best_val_rmse:
                        best_val_rmse, best_metrics = save_best_model(
                            model,
                            {'MAE': mae, 'RMSE': rmse, 'PCC': pcc, 'SNR': snr_pred, 'MACC': macc},
                            best_val_rmse,
                            f'best_model_{model_name}_fold_{fold + 1}_{data_name}.pth',
                        )
                        plot_hr_sp_bap(all_outputs, all_labels, f'{model_name}_sample_{fold+1}')
                        plot_bvp(all_outputs, all_labels, f'{model_name}_sample_{fold+1}')

                fold_results.append(best_metrics)
                mlflow.log_artifact(f'model_paths/best_model_{model_name}_fold_{fold + 1}_{data_name}.pth')
                input_sample = next(iter(train_dataloader))[0][:1].to(DEVICE)
                output_sample = model(input_sample)
                log_trained_model(model, model_name, input_sample, output_sample)

            average_results = {metric: np.mean([fold[metric] for fold in fold_results]) for metric in fold_results[0]}
            fold_results_dict[model_name] = fold_results
            average_results_dict[model_name] = average_results
            for metric, value in average_results.items():
                mlflow.log_metric(metric, value)

    return fold_results_dict, average_results_dict



def train_ampnet(data, labels, num_folds=4, num_epochs=40):
    model_name = 'AMPNet'
    fusion_dataloader = create_ampnet_dataloader(data, labels, batch_size=BATCH_SIZE)
    with mlflow.start_run(run_name=model_name, nested=False):
        mlflow.log_param('Model Name', model_name)
        mlflow.log_param('K-Folds', num_folds)
        fold_results = []
        dataloaders = create_custom_dataloaders(fusion_dataloader.dataset, batch_size=BATCH_SIZE, k=num_folds)

        for fold, (train_loader, val_loader) in enumerate(dataloaders):
            rgb_models, thermal_models = load_models(R_3EDSAN, T_3EDSAN, DEVICE)
            fusion_model = AMPNet(rgb_models, thermal_models, normalization=True).to(DEVICE)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(fusion_model.parameters(), lr=0.0001)
            best_val_rmse = float('inf')
            best_metrics = {}

            for epoch in range(num_epochs):
                train_loss = train_one_ampnet_epoch(fusion_model, train_loader, criterion, optimizer, DEVICE)
                val_loss, metrics, _, all_outputs, _, all_labels = evaluate_ampnet_model(fusion_model, val_loader, criterion, DEVICE, batch_size=BATCH_SIZE)
                mae, rmse, pcc, snr_pred, macc, _, _ = metrics
                print(f'AMPNet, Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
                if rmse < best_val_rmse:
                    best_val_rmse, best_metrics = save_best_model(
                        fusion_model,
                        {'MAE': mae, 'RMSE': rmse, 'PCC': pcc, 'SNR': snr_pred, 'MACC': macc},
                        best_val_rmse,
                        f'best_model_AMPNet_fold_{fold + 1}.pth',
                    )
                    plot_hr_sp_bap(all_outputs, all_labels, model_name='AMPNet')
                    plot_bvp(all_outputs, all_labels, model_name='AMPNet')

            fold_results.append(best_metrics)
            mlflow.log_artifact(f'model_paths/best_model_AMPNet_fold_{fold + 1}.pth')

        avg_metrics = {key: np.mean([result[key] for result in fold_results]) for key in fold_results[0]}
        for metric, value in avg_metrics.items():
            mlflow.log_metric(metric, value)
    return fold_results, avg_metrics


if __name__ == '__main__':
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT_NAME)

    videos, labels = prepare_base_data(TRAIN_FEATURES_PATH, TRAIN_LABELS_PATH, segment_length=SEGMENT_LENGTH)

    if MODALITY == 'rgb':
        videos = get_rgb_videos(videos)
        model_names = TRAIN_RGB_MODEL_NAMES
        model_classes = [MODEL_CLASS_MAP[name] for name in model_names]
        criterion_classes = [nn.MSELoss, nn.MSELoss, nn.MSELoss]
        fold_results, avg_results = train_unimodal(model_classes, model_names, criterion_classes, videos, labels, n_splits=K_FOLDS)
        print(tabulate(avg_results.items(), headers=['Model Name', 'Metrics']))
    elif MODALITY == 'thermal':
        videos = get_thermal_videos(videos)
        model_names = TRAIN_THERMAL_MODEL_NAMES
        model_classes = [MODEL_CLASS_MAP[name] for name in model_names]
        criterion_classes = [CosineSimilarityLoss, CosineSimilarityLoss, CosineSimilarityLoss]
        fold_results, avg_results = train_unimodal(model_classes, model_names, criterion_classes, videos, labels, n_splits=K_FOLDS)
        print(tabulate(avg_results.items(), headers=['Model Name', 'Metrics']))
    elif MODALITY == 'multimodal':
        fold_results, average_results = train_ampnet(videos, labels, num_folds=K_FOLDS)
        print(tabulate(average_results.items(), headers=['Metric', 'Average Value']))
    else:
        raise ValueError(f'Unsupported modality: {MODALITY}')
