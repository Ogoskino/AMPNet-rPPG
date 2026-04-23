import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocessing.preprocess import device

class temp_AM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(temp_AM, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        # Temporal Attention
        self.temporal_conv = nn.Conv3d(2, 1, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False)

    def temporal_attention(self, x):
        # Compute max pooling along channel (dim=1), height (dim=3), and width (dim=4)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max across channels
        max_out, _ = torch.max(max_out, dim=3, keepdim=True)  # Max across height
        max_out, _ = torch.max(max_out, dim=4, keepdim=True)  # Max across width

        # Compute mean pooling along channel (dim=1), height (dim=3), and width (dim=4)
        mean_out = torch.mean(x, dim=1, keepdim=True)  # Mean across channels
        mean_out = torch.mean(mean_out, dim=3, keepdim=True)  # Mean across height
        mean_out = torch.mean(mean_out, dim=4, keepdim=True)  # Mean across width

        # Concatenate max and mean along the channel dimension
        temporal_att = torch.cat([max_out, mean_out], dim=1)

        # Apply conv3d and sigmoid
        temporal_att = torch.sigmoid(self.temporal_conv(temporal_att))
        return x * temporal_att

    def forward(self, x):
        # Temporal attention
        x = self.temporal_attention(x)

        return x


class VideoCBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(VideoCBAM, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        # Channel Attention
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.fc2 = nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)

        # Spatial Attention
        self.spatial_conv = nn.Conv3d(2, 1, kernel_size=(1, 7, 7), padding=(0, 3, 3), bias=False)

    def channel_attention(self, x):
        # Adaptive Avg Pool
        avg_pool = F.adaptive_avg_pool3d(x, 1)
        # Adaptive Max Pool
        max_pool = F.adaptive_max_pool3d(x, 1)

        # Shared MLP for both avg_pool and max_pool
        avg_out = self.fc2(F.relu(self.fc1(avg_pool)))
        max_out = self.fc2(F.relu(self.fc1(max_pool)))

        # Sum outputs and apply sigmoid
        channel_att = torch.sigmoid(avg_out + max_out)
        return x * channel_att

    def spatial_attention(self, x):
        # Max Pool along channel and temporal dimensions
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max across channels
        max_out, _ = torch.max(max_out, dim=2, keepdim=True)  # Max across time

        # Mean Pool along channel and temporal dimensions
        mean_out = torch.mean(x, dim=1, keepdim=True)  # Mean across channels
        mean_out = torch.mean(mean_out, dim=2, keepdim=True)  # Mean across time

        # Concatenate along channel dimension
        spatial_att = torch.cat([max_out, mean_out], dim=1)

        # Pass through conv3d and sigmoid
        spatial_att = torch.sigmoid(self.spatial_conv(spatial_att))
        return x * spatial_att

    def forward(self, x):
        # Channel attention
        x = self.channel_attention(x)

        # Spatial attention
        x = self.spatial_attention(x)

        return x


# Define the 3EDSAN Model
class EDSAN(nn.Module):
    def __init__(self, frames=192, n_channels=3, model='RGB', is_cbam = True, is_tam = True):
        super(EDSAN, self).__init__()

        self.model = model
        self.is_cbam = is_cbam
        self.is_tam = is_tam

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(n_channels, 16, kernel_size=5, stride=1, padding=2),  # spatial encoding
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=5, stride=1, padding=2),  # spatial encoding
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=5, stride=1, padding=2),  # spatio-temporal encoding
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=[4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=[4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.conv_reduce = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1, stride=1, padding=0),  # spatio-temporal encoding
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock10 = nn.Conv3d(64, 1, kernel_size=1, stride=1, padding=0)

        self.MaxpoolSpaTem1 = nn.AvgPool3d((2, 2, 2), stride=2)
        self.MaxpoolSpaTem2 = nn.AvgPool3d((2, 4, 4), stride=[2, 4, 4])
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))
        self.cbam = VideoCBAM(in_channels=64)
        self.temp_AM = temp_AM(in_channels=64)

    def forward(self, xh):  # Input: Batch_size*[3, T, 128, 128]
        if self.model=='RGB':
            x = xh[:, :3, :, :, :]  # Shape: [batch_size, 3, frames, 128, 128]
        else:
            [_, channelx, _, _, _] = xh.shape
            if channelx == 1:
                x = xh
            else:
                x = xh[:, 3:, :, :, :]  # Shape: [batch_size, 3, frames, 128, 128]
        [batch, channel, length, width, height] = x.shape

        x = self.ConvBlock1(x)  # x [batch_size, 16, T, 128, 128]
        x = self.MaxpoolSpaTem1(x)  # x [batch_size, 16, T/2, 64, 64]
        x = self.ConvBlock2(x)  # x [batch_size, 32, T/2, 64, 64]
        x = self.MaxpoolSpaTem2(x)  # x [batch_size, 32, T/4, 32, 32]
        x = self.ConvBlock4(x)  # x [batch_size, 64, T/4, 32, 32]
        if self.is_cbam:
            x_cbam = self.cbam(x)
        else:
            x_cbam = x
        if self.is_tam:
            x_temp = self.temp_AM(x_cbam)
        else:
            x_temp = x_cbam
        if self.is_cbam and self.is_tam:
            x_temp = x_cbam * x_temp
        x = self.conv_reduce(x_temp)
        x = self.upsample(x)  # x [batch_size, 64, T/8, 16, 16]
        x = self.upsample2(x)  # x [batch_size, 64, T/4, 32, 32]
        x = self.poolspa(x)  # [batch_size, 64, frames, 1, 1]
        x = self.ConvBlock10(x)

        # Reshape for output
        rPPG = x.view(-1, length)  # Output shape [batch_size, frames]
        return rPPG
    
R_3EDSAN = EDSAN().to(device)
T_3EDSAN = EDSAN(n_channels=1, model='thermal').to(device)






if __name__ == "__main__":

    tensor1 = torch.rand(2, 4, 192, 128, 128)
    result1 = R_3EDSAN(tensor1)
    print(result1.shape)
    result2 = T_3EDSAN(tensor1)
    print(result2.shape)