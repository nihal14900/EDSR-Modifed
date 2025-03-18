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
            trunk.append(CrossAttentionBlock(in_channels=64, reduction=16, kernel_size=7))
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
# Spatial Attention Module
# ---------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """
        Computes spatial attention scores for input x (B, C, H, W).
        It concatenates the channel-wise average and maximum (B, 2, H, W),
        applies a convolution, and then a sigmoid.
        """
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        x_cat = torch.cat([avg_out, max_out], dim=1)     # (B, 2, H, W)
        out = self.conv(x_cat)
        return self.sigmoid(out)                         # (B, 1, H, W)

# ---------------------------
# Channel Attention Module
# ---------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        Computes channel attention scores for input x (B, C, H, W),
        returning an attention map of shape (B, C, 1, 1):
        A(x) = sigmoid(W2(ReLU(W1(GAP(x))))).
        """
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.gap(x)             # (B, C, 1, 1)
        out = self.fc1(avg)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out)      # (B, C, 1, 1)

# ---------------------------
# Cross-Attention Block (Spatial First, Channel Second)
# ---------------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        """
        Implements the block with:
         1. Two separate initial convolutions whose outputs are merged.
         2. Triple bond branch with dual conv paths and separate cross-spatial attention layers.
         3. Two separate intermediate convolutions (mid conv) whose outputs are merged.
         4. Double bond branch with dual conv paths and separate cross-channel attention layers.
         5. Residual skip connection.
        """
        super(CrossAttentionBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        
        # 1. Separate initial convolutions.
        self.conv_init1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.conv_init2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        
        # ---------------------------
        # 2. Triple Bond Branch (Cross-Spatial Attention)
        # ---------------------------
        # Two parallel convolutions.
        self.conv_trp1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.conv_trp2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        # Two separate spatial attention modules for cross-modulation.
        self.sa_12 = SpatialAttention(kernel_size)  # Modulate branch 1 with branch 2 features.
        self.sa_21 = SpatialAttention(kernel_size)  # Modulate branch 2 with branch 1 features.

        # 3. Separate intermediate (mid) convolutions.
        self.conv_mid1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.conv_mid2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        
        # ---------------------------
        # 4. Double Bond Branch (Cross-Channel Attention)
        # ---------------------------
        # Two parallel convolutions.
        self.conv_dbl1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.conv_dbl2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        # Two separate channel attention modules for cross-modulation.
        self.ca_12 = ChannelAttention(in_channels, reduction)  # Modulate branch 1 with branch 2 features.
        self.ca_21 = ChannelAttention(in_channels, reduction)  # Modulate branch 2 with branch 1 features.

    def forward(self, x):
        # 1. Initial Convolution: two separate paths.
        z0_1 = self.relu(self.conv_init1(x))
        z0_2 = self.relu(self.conv_init2(x))
        # Merge initial features.
        z0 = z0_1 + z0_2  # (B, C, H, W)
        
        # ---------------------------
        # 2. Triple Bond Branch (Cross-Spatial Attention)
        # ---------------------------
        # a. Parallel feature extraction.
        z1_trp = self.relu(self.conv_trp1(z0))
        z2_trp = self.relu(self.conv_trp2(z0))
        
        # b. Compute separate spatial attention scores.
        attn_from_z2_trp = self.sa_12(z2_trp)  # (B, 1, H, W) to modulate branch 1.
        attn_from_z1_trp = self.sa_21(z1_trp)  # (B, 1, H, W) to modulate branch 2.
        
        # c. Cross modulation.
        z1_trp_mod = z1_trp * attn_from_z2_trp
        z2_trp_mod = z2_trp * attn_from_z1_trp
        
        # d. Merge the two paths.
        M_trp = z1_trp_mod + z2_trp_mod
        
        # ---------------------------
        # 3. Intermediate (Mid) Convolution: two separate paths.
        # ---------------------------
        z1_mid = self.relu(self.conv_mid1(M_trp))
        z2_mid = self.relu(self.conv_mid2(M_trp))
        # Merge mid features.
        z1 = z1_mid + z2_mid  # (B, C, H, W)
        
        # ---------------------------
        # 4. Double Bond Branch (Cross-Channel Attention)
        # ---------------------------
        # a. Parallel feature extraction.
        z1_dbl = self.relu(self.conv_dbl1(z1))
        z2_dbl = self.relu(self.conv_dbl2(z1))
        
        # b. Compute separate channel attention scores.
        attn_from_z2 = self.ca_12(z2_dbl)  # (B, C, 1, 1) to modulate branch 1.
        attn_from_z1 = self.ca_21(z1_dbl)  # (B, C, 1, 1) to modulate branch 2.
        
        # c. Cross modulation.
        z1_dbl_mod = z1_dbl * attn_from_z2
        z2_dbl_mod = z2_dbl * attn_from_z1
        
        # d. Merge the two paths.
        M_dbl = z1_dbl_mod + z2_dbl_mod
        
        # ---------------------------
        # 5. Residual Skip Connection.
        # ---------------------------
        out = x + M_dbl
        return out

# ---------------------------
# Test Code
# ---------------------------
if __name__ == "__main__":
    # Define input dimensions.
    batch_size = 1
    channels = 64
    height = 32
    width = 32

    # Create a random input tensor.
    x = torch.randn(batch_size, channels, height, width)
    
    # Instantiate the cross-attention block.
    model = CrossAttentionBlock(in_channels=channels, reduction=16, kernel_size=7)
    print("CrossAttentionBlock architecture:")
    print(model)
    
    # Forward pass.
    out = model(x)
    
    # Print shapes to verify functionality.
    print("Input shape: ", x.shape)
    print("Output shape:", out.shape)
