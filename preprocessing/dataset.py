from torch.utils.data import Dataset, DataLoader

# Define RGB and Thermal models here (RGBModel and ThermalModel classes)

# Dataset definitions
class Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        return image, label

class AMPNet_Dataset(Dataset):
    def __init__(self, rgb_data, thermal_data, labels):
        """
        Args:
            rgb_data (list or np.ndarray): List or array of RGB video data.
            thermal_data (list or np.ndarray): List or array of Thermal video data.
            labels (list or np.ndarray): List or array of labels corresponding to each sample.
            transform_rgb (callable, optional): Optional transform to be applied on the RGB data.
            transform_thermal (callable, optional): Optional transform to be applied on the Thermal data.
        """
        self.rgb_data = rgb_data
        self.thermal_data = thermal_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        rgb_sample = self.rgb_data[idx]
        thermal_sample = self.thermal_data[idx]
        label = self.labels[idx]

        # Convert data and labels to tensors
        rgb_sample = rgb_sample.clone().detach().float()
        thermal_sample = thermal_sample.clone().detach().float()
        label = label.clone().detach().float()


        return rgb_sample, thermal_sample, label
    
def create_ampnet_dataloader(data, labels, batch_size=8, shuffle=True):
    # Create the fusion dataset
    rgb_data = data[:, :3, :, :, :]
    thermal_data = data[:, 3:, :, :, :]
    fusion_dataset = AMPNet_Dataset(rgb_data, thermal_data, labels)

    # Create the DataLoader
    fusion_dataloader = DataLoader(fusion_dataset, batch_size=batch_size, shuffle=shuffle)
    return fusion_dataloader