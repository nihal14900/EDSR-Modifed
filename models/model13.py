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
            trunk.append(DNA_SR_Block(in_channels=64, features=128))
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

# Large Kernel Attention (LKA) Module
class LKA(nn.Module):
    def __init__(self, channels):
        """
        Large Kernel Attention module with depthwise and dilated convolutions.

        Args:
        - channels (int): Number of input and output channels (same for all layers).
        """
        super(LKA, self).__init__()
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)  # Depthwise 5x5
        self.dw_d_conv = nn.Conv2d(channels, channels, kernel_size=7, padding=9, dilation=3, groups=channels)  # Dilated 7x7
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1)  # Pointwise 1x1

    def forward(self, x):
        """
        Forward pass through LKA.
        """
        return self.pw_conv(self.dw_d_conv(self.dw_conv(x)))

# Laplacian Edge Enhancement (LEE)
class LaplacianEdgeEnhance(nn.Module):
    def __init__(self):
        """
        Applies a Laplacian edge detection filter.
        """
        super(LaplacianEdgeEnhance, self).__init__()
        kernel = torch.tensor([[0, -1, 0], 
                               [-1, 4, -1], 
                               [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape (1,1,3,3)
        self.register_buffer('kernel', kernel)  # Persistent buffer (non-learnable)

    def forward(self, x):
        """
        Applies the Laplacian filter across all channels separately.
        """
        channels = x.shape[1]
        kernel = self.kernel.expand(channels, 1, 3, 3)  # Expand to match input channels
        return F.conv2d(x, kernel, padding=1, groups=channels)  # Depthwise filtering

# DNA-Inspired Super-Resolution Block
class DNA_SR_Block(nn.Module):
    def __init__(self, in_channels, features):
        """
        DNA-Inspired Super-Resolution Block.

        Args:
        - in_channels (int): Number of input channels.
        - features (int): Number of internal feature channels.
        """
        super(DNA_SR_Block, self).__init__()

        # First Path: Conv3 → Conv5 → Conv3 → Conv5
        self.conv3_1 = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(features, features, kernel_size=5, padding=2)
        self.conv3_2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(features, features, kernel_size=5, padding=2)

        # Second Path: Conv5 → Conv3 → Conv5 → Conv3
        self.conv5_a = nn.Conv2d(in_channels, features, kernel_size=5, padding=2)
        self.conv3_a = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv5_b = nn.Conv2d(features, features, kernel_size=5, padding=2)
        self.conv3_b = nn.Conv2d(features, features, kernel_size=3, padding=1)

        # Large Kernel Attention (LKA) for Dual Bond Interaction
        self.lka = LKA(features)
        self.alpha = nn.Parameter(torch.ones(1))  # Learnable weight for LKA scaling

        # Laplacian Edge Enhancement (LEE) for Triple Bond Interaction
        self.edge_enhance = LaplacianEdgeEnhance()
        self.lambda_edge = nn.Parameter(torch.ones(1))  # Learnable weight for edge enhancement

        # Group Normalization (after two interactions)
        self.group_norm = nn.GroupNorm(num_groups=8, num_channels=features)

        # Final Fusion Layer
        self.fusion_conv = nn.Conv2d(2 * features, in_channels, kernel_size=1)
        self.beta = nn.Parameter(torch.ones(1) * 0.5)  # Learnable fusion scaling

    def forward(self, x):
        """
        Forward pass of DNA_SR_Block.
        """
        # Parallel Pathway Processing
        x1 = self.conv3_1(x)
        x2 = self.conv5_a(x)

        x1 = F.gelu(x1)
        x2 = F.gelu(x2)

        # First Large Kernel Attention Interaction (Dual Bond)
        x1 = x1 + self.alpha * self.lka(x2)
        x2 = x2 + self.alpha * self.lka(x1)

        # Second Convolution Stage
        x1 = self.conv5_1(x1)
        x2 = self.conv3_a(x2)

        x1 = F.gelu(x1)
        x2 = F.gelu(x2)

        # Laplacian Edge Enhancement (Triple Bond)
        x1 = x1 + self.lambda_edge * self.edge_enhance(x2)
        x2 = x2 + self.lambda_edge * self.edge_enhance(x1)

        # Apply Group Normalization
        x1 = self.group_norm(x1)
        x2 = self.group_norm(x2)

        # Third Convolution Stage
        x1 = self.conv3_2(x1)
        x2 = self.conv5_b(x2)

        x1 = F.gelu(x1)
        x2 = F.gelu(x2)

        # Second Large Kernel Attention Interaction (Dual Bond)
        x1 = x1 + self.alpha * self.lka(x2)
        x2 = x2 + self.alpha * self.lka(x1)

        # Fourth Convolution Stage
        x1 = self.conv5_2(x1)
        x2 = self.conv3_b(x2)

        x1 = F.gelu(x1)
        x2 = F.gelu(x2)

        # Second Laplacian Edge Enhancement Interaction (Triple Bond)
        x1 = x1 + self.lambda_edge * self.edge_enhance(x2)
        x2 = x2 + self.lambda_edge * self.edge_enhance(x1)

        # Apply Group Normalization again
        x1 = self.group_norm(x1)
        x2 = self.group_norm(x2)

        # Final Fusion and Residual Connection
        fused = self.fusion_conv(torch.cat([x1, x2], dim=1))  # Channel concatenation
        out = x + self.beta * fused  # Residual connection
        
        return out

# Example Usage
if __name__ == "__main__":
    model = DNA_SR_Block(in_channels=64, features=128)  # 64 input channels, 128 feature channels
    x = torch.randn(1, 64, 128, 128)  # Example input tensor
    out = model(x)
    print("Output Shape:", out.shape)  # Expected: (1, 64, 128, 128)
