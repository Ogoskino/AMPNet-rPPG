

"""
RTrPPG

RTrPPG: An Ultra Light 3DCNN for Real-Time Remote Photoplethysmography
D. Botina-Monsalve et al."""

import torch.nn as nn
from preprocessing.preprocess import device

class N3DED64(nn.Module):
    def __init__(self, frames=192):  
        super(N3DED64, self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.Conv2 = nn.Sequential(
            nn.Conv3d(16, 32, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.Conv3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.Conv4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
       
        self.TrConv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.TrConv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock5 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        self.MaxpoolSpaTem_244_244 = nn.MaxPool3d((2, 4, 4), stride=(2,4,4))
        self.MaxpoolSpaTem_222_222 = nn.MaxPool3d((2, 2, 2), stride=2)
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x): # [batch, Features=3, Temp=128, Width=64, Height=64]
        # ENCODER
        x = x[:, :3, :, :, :]
        x = self.Conv1(x)		            # [b, F=3, T=128, W=64, H=64]->[b, F=16, T=128, W=64, H=64]
        x = self.MaxpoolSpaTem_222_222(x)   # [b, F=16, T=128, W=64, H=64]->[b, F=16, T=64, W=32, H=32]
        x = self.Conv2(x)		            # [b, F=16, T=64, W=32, H=32]->[b, F=32, T=64, W=32, H=32]
        x = self.MaxpoolSpaTem_244_244(x)   # [b, F=32, T=64, W=32, H=32]->[b, F=32, T=32, W=8, H=8]
        x = self.Conv3(x)		            # [b, F=32, T=32, W=8, H=8]->[b, F=64, T=32, W=8, H=8]
        
        x = self.Conv4(x)		            # [b, F=64, T=32, W=8, H=8]->[b, F=64, T=32, W=8, H=8]
        # DECODER
        x = self.TrConv1(x)		            # [b, F=64, T=32, W=8, H=8]->[b, F=64, T=64, W=8, H=8]
        x = self.TrConv2(x)		            # [b, F=64, T=64, W=8, H=8]->[b, F=64, T=128, W=8, H=8]     
        x = self.poolspa(x)                 # [b, F=64, T=128, W=8, H=8]->[b, F=64, T=128, W=1, H=1]
        x = self.ConvBlock5(x)             # [b, F=64, T=128, W=1, H=1]->[b, F=1, T=128, W=1, H=1]
        
        rPPG = x.view(-1,x.shape[2])        # [b,128]
        return rPPG
    



rtrppg = N3DED64().to(device)