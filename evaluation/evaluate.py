import os

import mlflow
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from torchinfo import summary

from evaluation.plots import (
    bland_altman_plot,
    plot_bvp_signals,
    plot_heart_rate,
    scatter_plot,
)
from signals.post_process import calculate_metric_per_video
from config import SESSION_LENGTH


FULL_SEQUENCE_LENGTH = SESSION_LENGTH

SPECIAL_LOSS_MODELS = {
    "RTrPPG-NPSNR",
    "R3EDSAN-NP",
    "RTrPPG",
    "R3EDSAN-NP-adLR",
    "PhysNet",
    "RTrPPG-NP",
    "TRTrPPG-NPSNR",
    "T3EDSAN-NP",
    "T3EDSAN-NP-adLR",
    "TPhysNet",
    "TRTrPPG-NP",
}


def _compute_loss(outputs, labels, criterion, model_name, device):
    if model_name in SPECIAL_LOSS_MODELS:
        seq_len = labels.shape[1]
        time_diff = (
            torch.linspace(0, 1, seq_len, device=device)
            .unsqueeze(0)
            .repeat(labels.shape[0], 1)
        )
        return criterion([outputs, labels, time_diff])

    return criterion(outputs, labels)


def _reconstruct_full_sequences(outputs, labels, full_sequence_length=FULL_SEQUENCE_LENGTH):
    """
    Rebuild full person sequences from segmented outputs.

    Input:
        outputs: (N_segments, segment_length)
        labels:  (N_segments, segment_length)

    Output:
        outputs_full: (N_sequences, full_sequence_length)
        labels_full:  (N_sequences, full_sequence_length)
    """
    outputs = np.asarray(outputs)
    labels = np.asarray(labels)

    if outputs.shape != labels.shape:
        raise ValueError(f"Shape mismatch: outputs {outputs.shape}, labels {labels.shape}")

    if outputs.ndim == 1:
        outputs = outputs[None, :]
        labels = labels[None, :]

    if outputs.ndim != 2:
        raise ValueError(f"Expected 2D arrays (N_segments, segment_length), got {outputs.shape}")

    flat_outputs = outputs.reshape(-1)
    flat_labels = labels.reshape(-1)

    if flat_outputs.size % full_sequence_length != 0:
        raise ValueError(
            f"Flattened output length {flat_outputs.size} is not divisible by full sequence length "
            f"{full_sequence_length}."
        )

    outputs_full = flat_outputs.reshape(-1, full_sequence_length)
    labels_full = flat_labels.reshape(-1, full_sequence_length)

    return outputs_full, labels_full


def evaluate_model(model, dataloader, criterion, device, model_name, sampling_rate=28):
    model.eval()
    val_loss = 0.0
    all_outputs, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = _compute_loss(outputs, labels, criterion, model_name, device)

            val_loss += loss.item() * inputs.size(0)
            all_outputs.append(outputs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    val_loss /= len(dataloader.dataset)

    all_outputs = np.concatenate(all_outputs, axis=0)   # (N_segments, 128)
    all_labels = np.concatenate(all_labels, axis=0)     # (N_segments, 128)

    metrics = compute_metrics(all_outputs, all_labels, sampling_rate=sampling_rate)
    return val_loss, metrics, all_outputs, all_labels


def compute_metrics(outputs, labels, sampling_rate=28, full_sequence_length=FULL_SEQUENCE_LENGTH):
    """
    Reconstruct full person sequences, then compute metrics per full sequence.
    """
    outputs_full, labels_full = _reconstruct_full_sequences(
        outputs, labels, full_sequence_length=full_sequence_length
    )

    hr_labels, hr_preds = [], []
    snrs, maccs, signals, time_lags = [], [], [], []

    for i in range(outputs_full.shape[0]):
        hr_true, hr_pred, snr, macc, signal, time_lag = calculate_metric_per_video(
            outputs_full[i],
            labels_full[i],
            fs=sampling_rate,
            diff_flag=True,
            use_bandpass=True,
            hr_method="FFT",
        )
        hr_labels.append(hr_true)
        hr_preds.append(hr_pred)
        snrs.append(snr)
        maccs.append(macc)
        signals.append(signal)
        time_lags.append(time_lag)

    hr_labels = np.asarray(hr_labels)
    hr_preds = np.asarray(hr_preds)
    signals = np.asarray(signals, dtype=object)

    mae = mean_absolute_error(hr_labels, hr_preds)
    rmse = float(np.sqrt(np.mean((hr_labels - hr_preds) ** 2)))

    if len(hr_labels) > 1 and np.std(hr_labels) > 0 and np.std(hr_preds) > 0:
        pcc = float(pearsonr(hr_labels, hr_preds)[0])
    else:
        pcc = 0.0

    snr_pred = float(np.mean(snrs))
    avg_macc = float(np.mean(maccs))
    avg_timelag = float(np.mean(time_lags))

    return mae, rmse, pcc, snr_pred, avg_macc, signals, avg_timelag


def plot_bvp(outputs, labels, model_name):
    outputs_full, labels_full = _reconstruct_full_sequences(outputs, labels)
    plot_bvp_signals(labels_full, outputs_full, model_name)


def plot_hr_sp_bap(outputs, labels, model_name, sampling_rate=28):
    outputs_full, labels_full = _reconstruct_full_sequences(outputs, labels)

    hr_labels, hr_preds = [], []

    for i in range(outputs_full.shape[0]):
        hr_true, hr_pred, _, _, _, _ = calculate_metric_per_video(
            outputs_full[i],
            labels_full[i],
            fs=sampling_rate,
            diff_flag=True,
            use_bandpass=True,
            hr_method="FFT",
        )
        hr_labels.append(hr_true)
        hr_preds.append(hr_pred)

    hr_labels = np.asarray(hr_labels)
    hr_preds = np.asarray(hr_preds)

    plot_heart_rate(hr_labels, hr_preds, model_name)
    scatter_plot(hr_labels, hr_preds, model_name)
    bland_altman_plot(hr_labels, hr_preds, model_name)


def log_and_visualize_results(
    all_outputs,
    metrics,
    all_labels,
    model_name,
    sampling_rate=28,
    fold=1,
):
    mae, rmse, pcc, snr_pred, macc, _, timelag = metrics

    plot_hr_sp_bap(all_outputs, all_labels, model_name, sampling_rate=sampling_rate)
    plot_bvp_signals(
        *_reconstruct_full_sequences(all_labels, all_outputs),
        model_name=model_name,
        macc=macc,
        timelag=timelag,
    )

    metrics_dict = {
        "MAE": mae,
        "RMSE": rmse,
        "PCC": pcc,
        "SNR_Pred": snr_pred,
        "MACC": macc,
    }

    print(model_name, metrics_dict)
    mlflow.log_metrics(metrics_dict)
    return metrics_dict


def save_best_model(model, metrics, best_rmse, filename, folder_path="model_paths"):
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, filename)

    if metrics["RMSE"] < best_rmse:
        torch.save(model.state_dict(), file_path)
        return metrics["RMSE"], metrics

    return best_rmse, metrics


def log_model_summary(model, model_name, input_size, device, folder_path="model_summaries"):
    os.makedirs(folder_path, exist_ok=True)

    model = model.to(device)
    summary_text = str(summary(model, input_size=input_size, device=device.type))

    summary_file_path = os.path.join(folder_path, f"{model_name}_summary.txt")
    with open(summary_file_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    mlflow.log_artifact(summary_file_path)
    print(f"Model summary saved and logged to MLflow: {summary_file_path}")


def evaluate_ampnet_model(fusion_model, dataloader, criterion, device, batch_size, sampling_rate=28):
    fusion_model.eval()
    val_loss = 0.0

    all_outputs, all_outputs_rgb, all_labels = [], [], []

    with torch.no_grad():
        for rgb_inputs, thermal_inputs, labels in dataloader:
            rgb_inputs = rgb_inputs.to(device)
            thermal_inputs = thermal_inputs.to(device)
            labels = labels.to(device)

            outputs, rgb_outputs, _ = fusion_model(rgb_inputs, thermal_inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * rgb_inputs.size(0)

            all_outputs.append(outputs.detach().cpu().numpy())
            all_outputs_rgb.append(rgb_outputs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    val_loss /= len(dataloader.dataset)

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_outputs_rgb = np.concatenate(all_outputs_rgb, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = compute_metrics(all_outputs, all_labels, sampling_rate=sampling_rate)
    metrics_rgb = compute_metrics(all_outputs_rgb, all_labels, sampling_rate=sampling_rate)

    return val_loss, metrics, metrics_rgb, all_outputs, all_outputs_rgb, all_labels