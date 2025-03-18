# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math

import torch
from torch import nn


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rcb(x)

        out = torch.mul(out, 0.1)
        out = torch.add(out, identity)

        return out


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.upsample_block(x)

        return out


class EDSR(nn.Module):
    def __init__(self, upscale_factor: int) -> None:
        super(EDSR, self).__init__()
        # First layer
        self.conv1 = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))

        # Residual blocks
        trunk = []
        for _ in range(16):
            trunk.append(DualPathAttentionBlock(in_channels=64, reduction=16, kernel_size=7))
        self.trunk = nn.Sequential(*trunk)

        # Second layer
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

        # Upsampling layers
        upsampling = []
        if upscale_factor == 2 or upscale_factor == 4:
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling.append(UpsampleBlock(64, 2))
        elif upscale_factor == 3:
            upsampling.append(UpsampleBlock(64, 3))
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))

        self.register_buffer("mean", torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # The images by subtracting the mean RGB value of the DIV2K dataset.
        out = x.sub_(self.mean).mul_(255.)

        out1 = self.conv1(out)
        out = self.trunk(out1)
        out = self.conv2(out)
        out = torch.add(out, out1)
        out = self.upsampling(out)
        out = self.conv3(out)

        out = out.div_(255.).add_(self.mean)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)


import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Spatial Attention Module (Cross-Modulating)
# ---------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """
        Computes spatial attention scores and applies cross-modulation.
        Uses channel-wise max pooling and average pooling, followed by a conv layer.
        """
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn_map = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn_map  # Modulate input x using computed spatial attention

# ---------------------------
# Channel Attention Module (Cross-Modulating)
# ---------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        Computes channel attention scores and applies cross-modulation.
        Uses Global Average Pooling followed by two FC layers.
        """
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn_map = self.sigmoid(self.fc2(self.relu(self.fc1(self.gap(x)))))
        return x * attn_map  # Modulate input x using computed channel attention

# ---------------------------
# Dual Convolutional Pathway
# ---------------------------
class DualConvPath(nn.Module):
    def __init__(self, in_channels):
        """
        Applies two parallel convolution paths and merges the results.
        """
        super(DualConvPath, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        path1 = self.relu(self.conv1(x))
        path2 = self.relu(self.conv2(x))
        return path1 + path2  # Merge two pathways

# ---------------------------
# Cross-Spatial Attention Block
# ---------------------------
class CrossSpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        """
        Applies spatial attention on input features.
        """
        super(CrossSpatialAttentionBlock, self).__init__()
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(x)

# ---------------------------
# Cross-Channel Attention Block
# ---------------------------
class CrossChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        Applies channel attention on input features.
        """
        super(CrossChannelAttentionBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)

    def forward(self, x):
        return self.channel_attention(x)

# ---------------------------
# Dual-Pathway Block with Spatial-First, Channel-Second Cross-Attention
# ---------------------------
class DualPathAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        """
        Implements:
         1. Dual-path feature extraction.
         2. Spatial attention first (Triple Bond Branch).
         3. Intermediate dual-path refinement.
         4. Channel attention second (Double Bond Branch).
         5. Residual skip connection.
        """
        super(DualPathAttentionBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        # Initial Dual-Path Feature Extraction
        self.init_dual_conv = DualConvPath(in_channels)

        # Spatial Attention (Triple Bond)
        self.spatial_attention = CrossSpatialAttentionBlock(in_channels, kernel_size)

        # Intermediate Dual-Path Refinement
        self.mid_dual_conv = DualConvPath(in_channels)

        # Channel Attention (Double Bond)
        self.channel_attention = CrossChannelAttentionBlock(in_channels, reduction)

    def forward(self, x):
        identity = x  # Preserve input for residual connection

        # 1. Initial Feature Extraction
        x = self.init_dual_conv(x)

        # 2. Spatial Attention (Triple Bond)
        x = self.spatial_attention(x)

        # 3. Intermediate Refinement
        x = self.mid_dual_conv(x)

        # 4. Channel Attention (Double Bond)
        x = self.channel_attention(x)

        # 5. Residual Skip Connection
        return identity + x  # Residual learning

# ---------------------------
# Test Code
# ---------------------------
if __name__ == "__main__":
    # Define input dimensions
    batch_size = 1
    channels = 64
    height = 32
    width = 32

    # Create a random input tensor
    x = torch.randn(batch_size, channels, height, width)

    # Instantiate the dual-path attention block
    model = DualPathAttentionBlock(in_channels=channels, reduction=16, kernel_size=7)
    print("DualPathAttentionBlock architecture:")
    print(model)

    # Forward pass
    out = model(x)

    # Print shapes to verify functionality
    print("Input shape: ", x.shape)
    print("Output shape:", out.shape)
