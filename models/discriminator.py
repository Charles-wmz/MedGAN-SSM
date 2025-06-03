import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """PatchGAN-based Discriminator"""
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, stride=1, padding=1))  # Reduced one downsampling layer, simplified discriminator

    def forward(self, x):
        return self.model(x) 