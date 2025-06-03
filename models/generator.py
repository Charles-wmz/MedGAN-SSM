import torch
import torch.nn as nn

class StateSpaceModule(nn.Module):
    """State Space Construction Module"""
    def __init__(self, in_channels, state_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, state_dim, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, state_dim)
        
    def forward(self, x):
        # Basic state mapping
        s = torch.relu(self.conv(x))
        
        # Global context fusion
        g = self.gap(x).squeeze(-1).squeeze(-1)
        g = torch.relu(self.fc(g))[:, :, None, None]
        
        return s + g  # Modulated state Ŝ

class DynamicGate(nn.Module):
    """Dynamic Gating Module"""
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        # Spatial gating branch
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//ratio, 3, padding=1),
            nn.ReLU(),
            NonLocalBlock(in_channels//ratio),
            nn.Conv2d(in_channels//ratio, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Channel gating branch
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//ratio, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//ratio, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        g_spatial = self.spatial(x)
        g_channel = self.channel(x)
        return g_spatial * 0.6 + g_channel * 0.4  # Weighted fusion

class MultiScaleS6(nn.Module):
    """Multi-scale S6 Module"""
    def __init__(self, channels):
        super().__init__()
        # Local branch
        self.local = nn.Sequential(
            ResidualBlock(channels, dilation=1),
            ResidualBlock(channels, dilation=2),
            ResidualBlock(channels, dilation=4)
        )
        
        # Global branch
        self.global_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 7, padding=3),
            NonLocalBlock(channels)
        )
        
    def forward(self, x, gate):
        local_feat = self.local(x)
        global_feat = self.global_branch(x)
        return x + gate*local_feat + (1-gate)*global_feat

class ResidualBlock(nn.Module):
    """Residual Block"""
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.InstanceNorm2d(channels))
        
    def forward(self, x):
        return x + self.conv(x)

class NonLocalBlock(nn.Module):
    """Non-local Attention Module"""
    def __init__(self, channels):
        super().__init__()
        self.theta = nn.Conv2d(channels, channels//8, 1)
        self.phi = nn.Conv2d(channels, channels//8, 1)
        self.g = nn.Conv2d(channels, channels//8, 1)
        self.out = nn.Conv2d(channels//8, channels, 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        theta = self.theta(x).view(batch_size, -1, x.size(2)*x.size(3))
        phi = self.phi(x).view(batch_size, -1, x.size(2)*x.size(3)).permute(0,2,1)
        g = self.g(x).view(batch_size, -1, x.size(2)*x.size(3))
        
        attn = torch.softmax(torch.bmm(theta, phi), dim=-1)
        out = torch.bmm(attn, g)
        out = out.view(batch_size, -1, x.size(2), x.size(3))
        return self.out(out) + x

class Generator(nn.Module):
    """
    2D Image Generator - Processing Multi-modal MRI Slices
    Input: 2D multi-modal MRI slices (variable channels, depending on available modalities)
    Output: Single modality 2D MRI slice
    """
    def __init__(self, in_channels=None, out_channels=1):
        super().__init__()
        if in_channels is None:
            in_channels = 3
            
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            StateSpaceModule(64, 64)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        # Core processing modules
        self.process = nn.Sequential(
            MultiScaleS6(512),
            DynamicGate(512),
            MultiScaleS6(512)
        )
        
        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1),  # 512 = 256 + 256 (skip connection)
            nn.InstanceNorm2d(128),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1),  # 256 = 128 + 128 (skip connection)
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),  # 128 = 64 + 64 (skip connection)
            nn.Tanh()
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        gate = self.process[1](enc4)
        proc = self.process[0](enc4, gate)
        proc = self.process[2](proc, gate)

        dec1 = self.decoder1(proc)
        dec2 = self.decoder2(torch.cat([dec1, enc3], dim=1))  # Skip connection
        dec3 = self.decoder3(torch.cat([dec2, enc2], dim=1))  # Skip connection
        dec4 = self.decoder4(torch.cat([dec3, enc1], dim=1))  # Skip connection

        return dec4 