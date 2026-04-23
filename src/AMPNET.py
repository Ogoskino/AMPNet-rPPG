import torch
import torch.nn as nn

class AMPNet(nn.Module):
    def __init__(self, rgb_models, thermal_models, normalization=True):
        super(AMPNet, self).__init__()
        self.rgb_models = rgb_models  # List of n number of R-3EDSAN models, one for each fold
        self.thermal_models = thermal_models  # List of n number of T-3EDSAN models, one for each fold   

        # Define normalization layers (if required)
        if normalization:
            self.rgb_norm = nn.LayerNorm(192)  # Assuming frames = 192
            self.thermal_norm = nn.LayerNorm(192)  # Assuming frames = 192
        else:
            self.rgb_norm = self.thermal_norm = None

        # Define additional layers for fusion
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=1),  # Concatenate the outputs of RGB and Thermal
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb_input, thermal_input):
        # Pass the RGB input through all RGB models and average the output
        rgb_outputs = []
        for rgb_model in self.rgb_models:
            rgb_output = rgb_model(rgb_input)  # Expected shape: [batch_size, frames]
            rgb_outputs.append(rgb_output.unsqueeze(0))  # Add a new dimension for stacking
        avg_rgb_output = torch.mean(torch.cat(rgb_outputs, dim=0), dim=0)  # Average along the first dimension

        # Pass the Thermal input through all Thermal models and average the output
        thermal_outputs = []
        for thermal_model in self.thermal_models:
            thermal_output = thermal_model(thermal_input)  # Expected shape: [batch_size, frames]
            thermal_outputs.append(thermal_output.unsqueeze(0))  # Add a new dimension for stacking
        avg_thermal_output = torch.mean(torch.cat(thermal_outputs, dim=0), dim=0)  # Average along the first dimension

        # Apply normalization if enabled
        if self.rgb_norm:
            avg_rgb_output_n = self.rgb_norm(avg_rgb_output)  # Normalize RGB output
            avg_th_output = self.thermal_norm(avg_thermal_output)  # Normalize Thermal output

        # Concatenate the averaged RGB and Thermal outputs
        combined_output = torch.cat((avg_rgb_output_n.unsqueeze(1), avg_th_output.unsqueeze(1)), dim=1)  

        # Pass through the fusion layer
        fused_output = self.fusion_layer(combined_output)
        fused_output = fused_output.view(-1, combined_output.shape[2])  # Shape: [batch_size, frames]

        return fused_output, avg_rgb_output, avg_thermal_output  # Output shape: [batch_size, frames]
    

# Function to load all pre-trained models for each fold
def load_models(rgb_model, thermal_model, device):
    # Load all four RGB and Thermal models for the current fold
    rgb_models = []
    thermal_models = []

    rgb_model_path = 'model_paths/R3EDSAN.pth'
    rgb_model = rgb_model
    rgb_model.load_state_dict(torch.load(rgb_model_path, weights_only=True, map_location=device))
    rgb_model.eval()
    rgb_models.append(rgb_model)

    thermal_model_path = 'model_paths/T3EDSAN.pth'
    thermal_model = thermal_model
    thermal_model.load_state_dict(torch.load(thermal_model_path, weights_only=True, map_location=device))
    thermal_model.eval()
    thermal_models.append(thermal_model)

    return rgb_models, thermal_models

