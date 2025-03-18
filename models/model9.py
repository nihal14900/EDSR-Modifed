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
            trunk.append(EnhancedCrossAttentionBlock(in_channels=64, reduction=16, kernel_size=7))
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
# CBAM Attention Module (Combined Spatial & Channel Attention)
# ---------------------------
class CBAMAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAMAttention, self).__init__()
        # Channel Attention
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid_ch = nn.Sigmoid()
        
        # Spatial Attention
        self.conv_sp = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid_sp = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.gap(x)
        ch_attn = self.sigmoid_ch(self.fc2(self.relu(self.fc1(avg_out))))
        x = x * ch_attn  # Apply channel attention

        # Spatial Attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        sp_attn = self.sigmoid_sp(self.conv_sp(torch.cat([avg_pool, max_pool], dim=1)))
        x = x * sp_attn  # Apply spatial attention

        return x

# ---------------------------
# Residual Bottleneck Block
# ---------------------------
class ResidualBottleneck(nn.Module):
    def __init__(self, in_channels, expansion=4):
        super(ResidualBottleneck, self).__init__()
        hidden_dim = in_channels // expansion
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return identity + out  # Residual Connection

# ---------------------------
# Multi-Scale Feature Fusion
# ---------------------------
class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleFusion, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, bias=False)
        self.conv7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, bias=False)
        self.fusion_weight = nn.Parameter(torch.ones(4))  # Learnable fusion weights

    def forward(self, x):
        feat1 = self.conv1x1(x)
        feat2 = self.conv3x3(x)
        feat3 = self.conv5x5(x)
        feat4 = self.conv7x7(x)
        # Learnable fusion
        fused = self.fusion_weight[0] * feat1 + self.fusion_weight[1] * feat2 + \
                self.fusion_weight[2] * feat3 + self.fusion_weight[3] * feat4
        return fused

# ---------------------------
# Enhanced Cross-Attention Block
# ---------------------------
class EnhancedCrossAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(EnhancedCrossAttentionBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        
        # Initial Feature Extraction
        self.init_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)

        # Multi-Scale Feature Fusion
        self.ms_fusion = MultiScaleFusion(in_channels)

        # Residual Bottleneck Block
        self.res_bottleneck = ResidualBottleneck(in_channels)

        # CBAM Attention (Spatial + Channel)
        self.cbam_attn = CBAMAttention(in_channels, reduction, kernel_size)

        # Final Convolution
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        identity = x  # Save input for residual connection

        # Initial feature extraction
        x = self.relu(self.init_conv(x))

        # Multi-Scale Feature Fusion
        x = self.ms_fusion(x)

        # Residual Bottleneck Block
        x = self.res_bottleneck(x)

        # CBAM Attention
        x = self.cbam_attn(x)

        # Final convolution for refinement
        x = self.relu(self.final_conv(x))

        # Residual Connection
        return identity + x  # Skip Connection

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
    
    # Instantiate the improved cross-attention block
    model = EnhancedCrossAttentionBlock(in_channels=channels, reduction=16, kernel_size=7)
    print("EnhancedCrossAttentionBlock architecture:")
    print(model)
    
    # Forward pass
    out = model(x)
    
    # Print shapes to verify functionality
    print("Input shape: ", x.shape)
    print("Output shape:", out.shape)
