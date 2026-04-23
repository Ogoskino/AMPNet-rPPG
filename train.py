from src.iBVPNet import iBVPNet
from src.EDSAN import EDSAN, R_3EDSAN, T_3EDSAN
from src.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from src.RTrPPG import N3DED64
from evaluate.loss import *
import mlflow
from mlflow.models.signature import infer_signature
import numpy as np
import torch
import os
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from tabulate import tabulate
import torch.optim as optim
from preprocessing.dataset import *
from evaluate.evaluate import *
from preprocessing.preprocess import device
from preprocessing.dataloader import load_iBVP_dataset
from preprocessing.preprocess import *
from src.AMPNET import *
from torchinfo import summary



def log_model_summary(model, model_name, input_size, device, folder_path="model_summaries"):
    """
    Logs the summary of the model to MLflow and saves it in a specified folder.
    
    Args:
        model: The PyTorch model instance.
        model_name: The name of the model (e.g., "RGB", "Thermal").
        input_size: The input size tuple to the model.
        device: The device (e.g., "cpu" or "cuda").
        folder_path: The folder path where the summary file will be saved (default: "model_summaries").
    """
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    model = model.to(device)
    
    # Get model summary
    summary_text = str(summary(model, input_size=input_size, device=device.type))

    # Save model summary as a text file in the folder
    summary_file_path = os.path.join(folder_path, f"{model_name}_summary.txt")
    
    with open(summary_file_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    
    # Log the summary file as an artifact in MLflow
    mlflow.log_artifact(summary_file_path)
    
    print(f"Model summary saved and logged to MLflow: {summary_file_path}")


def log_trained_model(model, model_name, input_sample, output_sample, device):
    # Ensure output_sample is detached before converting to numpy
    output_sample_np = output_sample.detach().cpu().numpy()
    input_sample_np = input_sample.cpu().numpy()

    signature = infer_signature(input_sample_np, output_sample_np)
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path=model_name,
        input_example=input_sample_np,
        signature=signature,
    )


def initialize_model(model_class, model_name, criterion_classes, optimizer_classes, scheduler_classes, model_idx, device):
    """Initializes model, criterion, optimizer, and scheduler."""
    if model_name == 'iBVPNet':
        model = model_class(in_channels=3, frames=192, debug=True).to(device)
    if model_name == 'Thermal':
        model = model_class(n_channels=1, model='thermal').to(device)
    else:
        model = model_class().to(device)

    if model_name == 'RTrPPG':
        criterion = criterion_classes[model_idx](Lambda=1.32)
        optimizer = optimizer_classes[model_idx](model.parameters(), lr=0.00044)
    else:
        criterion = criterion_classes[model_idx]()
        optimizer = optimizer_classes[model_idx](model.parameters(), lr=0.0001)

    scheduler = scheduler_classes[model_idx]
    if scheduler:
        scheduler = scheduler(optimizer, step_size=2, gamma=0.95)
    
    return model, criterion, optimizer, scheduler



def initialize_ampnet(rgb_model_class, thermal_model_class, device):
    """Initializes model, criterion, and optimizer."""
    
    # Load models for this fold
    rgb_models, thermal_models = load_models(rgb_model_class, thermal_model_class, device)

    # Initialize the Fusion Model with the models for the current fold
    fusion_model = AMPNet(rgb_models, thermal_models, normalization=True).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)
    return fusion_model, criterion, optimizer



def train_one_epoch(model, dataloader, criterion, optimizer, device, model_name, batch_size):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        time_diff = torch.linspace(0, 1, inputs.shape[1]).unsqueeze(0).repeat(batch_size, 1).to(device)
        loss = criterion([outputs, labels, time_diff]) if model_name == 'RTrPPG' else criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)

def train_one_ampnet_epoch(fusion_model, train_loader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    fusion_model.train()
    running_loss = 0.0

    for rgb_inputs, thermal_inputs, labels in train_loader:
        rgb_inputs = rgb_inputs.to(device)
        thermal_inputs = thermal_inputs.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs, _, _ = fusion_model(rgb_inputs, thermal_inputs)

        # Compute loss and backpropagate
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * rgb_inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def create_custom_folds(dataset_length):
    """Create custom folds based on skin tone groups."""
    num_samples_per_skin_tone = dataset_length // 4
    indices_A = np.arange(0, num_samples_per_skin_tone)
    indices_B = np.arange(num_samples_per_skin_tone, 2 * num_samples_per_skin_tone)
    indices_C = np.arange(2 * num_samples_per_skin_tone, 3 * num_samples_per_skin_tone)
    indices_D = np.arange(3 * num_samples_per_skin_tone, 4 * num_samples_per_skin_tone)

    folds = [
        (np.hstack([indices_A, indices_B, indices_C]), indices_D),
        (np.hstack([indices_A, indices_B, indices_D]), indices_C),
        (np.hstack([indices_A, indices_C, indices_D]), indices_B),
        (np.hstack([indices_B, indices_C, indices_D]), indices_A)
    ]
    return folds


def create_custom_dataloaders(dataset, batch_size, k=4, mode="train"):
    """Creates custom DataLoaders for the given dataset based on skin tone groups."""
    dataset_length = len(dataset)
    folds = create_custom_folds(dataset_length)  # Create the custom folds
    
    dataloaders = []

    for i, (train_idx, val_idx) in enumerate(folds):
        if mode == "train":
            # Create train and validation subsets using the indices from the custom folds
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            # Create DataLoaders for the current fold
            train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            dataloaders.append((train_dataloader, val_dataloader))
        else:
            val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            if i == 0:
                dataloaders.append((None, val_dataloader))  # No train loader in eval mode

    return dataloaders

def create_kfold_dataloaders(dataset, batch_size, k=5, mode="train"):
    """Creates K-fold DataLoaders for the given dataset.
    
    Args:
        dataset (Dataset): The dataset to split.
        batch_size (int): The batch size for the DataLoader.
        k (int): The number of splits (default is 5).
        mode (str): The mode for creating DataLoaders. If "train", create train/validation splits; 
                    If not "train" (e.g., "eval"), use the entire dataset for evaluation.
    
    Returns:
        list: A list of tuples containing the DataLoaders for each fold.
    """
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    dataloaders = []

    for i, (train_idx, val_idx) in enumerate(kfold.split(dataset)):

        if mode == "train":

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            # Create DataLoaders for training and validation
            train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            dataloaders.append((train_dataloader, val_dataloader))
        else:
            # Use the entire dataset for evaluation
            val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            if i == 0:
                dataloaders.append((None, val_dataloader))  # No train loader in eval mode

    return dataloaders


def train_and_evaluate(models, n_splits, model_names, device, data, label):
    """Performs k-fold cross-validation for multiple models."""
    fold_results_dict = {name: [] for name in model_names}
    average_results_dict = {name: {} for name in model_names}
    
    for model_idx, model_class in enumerate(models):
        model_name = model_names[model_idx]
        dataset = Dataset(data=data, labels=label)
        print(f"\nTraining Model {model_name} with {n_splits}-fold Cross-Validation")

        with mlflow.start_run(run_name=model_name, nested=True):
            mlflow.log_param("Model Name", model_name)
            mlflow.log_param("K-Folds", n_splits)

            fold_results = [] 

            # Create K-fold DataLoaders
            dataloaders = create_kfold_dataloaders(dataset, batch_size=8, k=n_splits)

            for fold, (train_dataloader, val_dataloader) in enumerate(dataloaders):
                print(f"\nTraining Model {model_name}, Fold {fold + 1}/{n_splits}")

                # Initialize model, criterion, optimizer, and scheduler
                model, criterion, optimizer, scheduler = initialize_model(
                    model_class, model_name, criterion_classes, optimizer_classes, scheduler_classes, model_idx, device
                )

                # Log the model summary on the first fold
                input_size = (8, 1, 192, 64, 64) if model_name == 'Thermal' else (8, 3, 192, 64, 64)
                if fold == 0:  # Log summary for the first fold
                    log_model_summary(model, model_name, input_size, device)

                best_val_rmse = float('inf')
                best_metrics = {}

                for epoch in range(1):  # Adjust the number of epochs as needed
                    # Training step
                    train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device, model_name, batch_size=8)

                    # Evaluation step
                    val_loss, metrics, all_outputs, all_labels = evaluate_model(model, val_dataloader, criterion, device, model_name)

                    # Compute metrics
                    #metrics = compute_metrics(all_outputs, all_labels)
                    mae, rmse, pcc, snr_pred, macc = metrics

                    if scheduler:
                        scheduler.step()

                    print(f"Model {model_name}, Fold {fold + 1}, Epoch {epoch + 1}, "
                          f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

                    # Save best model based on RMSE
                    if rmse < best_val_rmse:
                        best_val_rmse, best_metrics = save_best_model(
                            model, {'MAE': mae, 'RMSE': rmse, 'PCC': pcc, 'SNR_Pred': snr_pred, 'MACC': macc},
                            best_val_rmse, f'best_model_{model_name}_fold_{fold + 1}.pth'
                        )
                        plot_hr(all_outputs, all_labels, model_name, sampling_rate=28)
                        plot_bvp(all_outputs, all_labels, model_name)

                fold_results.append(best_metrics)

                # Log the best model as an artifact
                mlflow.log_artifact(f'model_paths/best_model_{model_name}_fold_{fold + 1}.pth')
                input_sample = next(iter(train_dataloader))[0][:1].to(device)  # Single batch, single example
                output_sample = model(input_sample)

                # Log the trained model
                log_trained_model(model, model_name, input_sample, output_sample, device)

            # Aggregate fold results for the current model
            average_results = {metric: np.mean([fold[metric] for fold in fold_results]) for metric in fold_results[0]}
            fold_results_dict[model_name] = fold_results
            average_results_dict[model_name] = average_results

            # Log average results for the model
            for metric, value in average_results.items():
                mlflow.log_metric(metric, value)

    return fold_results_dict, average_results_dict



def train_AMPNet(fusion_dataloader, num_folds=2, num_epochs=1, device=torch.device('cuda')):
    """Trains and evaluates AMPNet with k-fold cross-validation."""
    with mlflow.start_run(run_name=model_name, nested=False):
        mlflow.log_param("Model Name", model_name)
        mlflow.log_param("K-Folds", num_folds)
    
        fold_results, fold_results_rgb = [], []
        
        # Create K-fold DataLoaders using the provided function
        dataloaders = create_kfold_dataloaders(fusion_dataloader.dataset, batch_size=8, k=num_folds)
        
        for fold, (train_loader, val_loader) in enumerate(dataloaders):
            print(f"Fold {fold + 1}/{num_folds}")

            # Initialize models and fusion model
            rgb_models, thermal_models = load_models(R_3EDSAN, T_3EDSAN, device)
            fusion_model = AMPNet(rgb_models, thermal_models, normalization=True).to(device)
            # Log the model summary on the first fold
            input_size = (8, 4, 192, 64, 64)
            if fold == 0:  # Log summary for the first fold
                #log_model_summary(fusion_model, "AMPNet", input_size, device)
                print("past_summary")

            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)

            best_val_rmse = float('inf')
            best_val_rmse_rgb = float('inf')
            best_metrics, best_metrics_rgb = {}, {}

            for epoch in range(num_epochs):
                # Train for one epoch
                train_loss = train_one_ampnet_epoch(fusion_model, train_loader, criterion, optimizer, device)
                print(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")

                # Evaluate on validation set
                val_loss, metrics, metrics_rgb, all_outputs, all_outputs_rgb, all_labels = evaluate_ampnet_model(fusion_model, val_loader, criterion, device)

                print(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}")
                print(f"Metrics: {metrics}, Metrics_RGB: {metrics_rgb}")

                mae, rmse, pcc, snr_pred, macc = metrics
                print(f"AMPNet, Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                if rmse < best_val_rmse:
                    best_val_rmse, best_metrics = save_best_model(
                        fusion_model, {'MAE': mae, 'RMSE': rmse, 'PCC': pcc, 'SNR_Pred': snr_pred, 'MACC': macc},
                        best_val_rmse, f'best_model_AMPNet_fold_{fold + 1}.pth'
                    )
                    print(f"Fold {fold + 1}: Best model saved with Validation RMSE: {best_val_rmse:.4f}")
                    plot_hr(all_outputs, all_labels, model_name="AMPNet", sampling_rate=28)
                    plot_bvp(all_outputs, all_labels, model_name="AMPNet")
                
                mae, rmse, pcc, snr_pred, macc = metrics_rgb
                print(f"R-3EDSAN, Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                if rmse < best_val_rmse_rgb:
                    best_val_rmse_rgb, best_metrics_rgb = save_best_model(
                        R_3EDSAN, {'MAE': mae, 'RMSE': rmse, 'PCC': pcc, 'SNR_Pred': snr_pred, 'MACC': macc},
                        best_val_rmse_rgb, f'best_model_RGB_fold_{fold + 1}.pth'
                    )
                    print(f"Fold {fold + 1}: Best model saved with Validation RMSE: {best_val_rmse_rgb:.4f}")
                    plot_hr(all_outputs_rgb, all_labels, model_name="R-3EDSAN", sampling_rate=28)
                    plot_bvp(all_outputs_rgb, all_labels, model_name="R-3EDSAN")

            # Save fold results
            fold_results.append(best_metrics)
            fold_results_rgb.append(best_metrics_rgb)
            mlflow.log_artifact(f'model_paths/best_model_{model_name}_fold_{fold + 1}.pth')

        # Compute average metrics across folds
        avg_metrics = {key: np.mean([result[key] for result in fold_results]) for key in fold_results[0]}
        avg_metrics_rgb = {key: np.mean([result[key] for result in fold_results_rgb]) for key in fold_results_rgb[0]}
        print(f"Average Metrics: {avg_metrics}")
        for metric, value in avg_metrics.items():
            mlflow.log_metric(metric, value)
        #print(f"Average Metrics RGB: {avg_metrics_rgb}")

    return fold_results, avg_metrics, fold_results_rgb, avg_metrics_rgb


if __name__ == "__main__":

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    #mlflow.set_experiment("rppg_project")
    model_classes = [EDSAN, EDSAN, iBVPNet, PhysNet_padding_Encoder_Decoder_MAX, N3DED64]  # Model classes instead of instances
    criterion_classes = [nn.MSELoss, nn.MSELoss, CosineSimilarityLoss, nn.MSELoss, NPSNR]
    optimizer_classes = [optim.Adam, optim.Adam, optim.Adam, optim.Adam, optim.Adam]
    scheduler_classes = [None, None,optim.lr_scheduler.StepLR, None, None]  # Optional
    model_names = ['RGB', 'Thermal', 'iBVPNet', 'PhysNet','RTrPPG']

    #dataset_path = r'C:\Users\jkogo\OneDrive\Desktop\PHD resources\datasets\iBVP_Dataset'
    #rgb_face, thermal_face, label = load_iBVP_dataset(dataset_path)
    #a, b = preprocess_iBVP_data(rgb_face, thermal_face, label)
    #video_chunks, label_chunks = extract_segments(a, b)
    #print(video_chunks.shape)  # Should be [num_chunks, chunk_size, height, width, channels]
    #print(label_chunks.shape)  # Should be [num_chunks, chunk_size]

    videos = torch.rand(56, 4, 192, 64, 64)
    label = torch.rand(56, 192)


    fold_results, avg_results = train_and_evaluate(
        model_classes, n_splits=2, model_names=model_names, device=device,data=videos, label=label
    )
    #Print Results
    print("\nSummary of Results Across All Models and Folds:")
    for model_name in model_names:
        print(f"\nResults for {model_name}:")
        print(tabulate(fold_results[model_name], headers="keys"))

    print("\nAverage Results for Each Model Across All Folds:")
    print(tabulate(avg_results.items(), headers=["Model Name", "Metrics"]))

    
    model_name = "AMPNet"
    fusion_dataloader = create_ampnet_dataloader(videos, label, batch_size=8)
    # Train the fusion model using the defined fusion_dataloader
    print("start")
    fold_results, average_results, fold_results_rgb, average_results_rgb = train_AMPNet(fusion_dataloader, num_folds=2, num_epochs=2, device=device)
    # Print the tabulated results for fold results
    print("\nSummary of Results Across All Folds:")
    print(tabulate(fold_results, headers="keys"))
    print(tabulate(average_results.items(), headers=["Metric", "Average Value"], tablefmt="grid"))





    
