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
            # trunk.append(ResidualConvBlock(64))
            trunk.append(DNAInteractionBlock(channels=64))
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

class DNAInteractionBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        """
        DNA-Inspired Dual-Path Interaction Block for Super-Resolution

        Args:
            channels (int): Number of input/output feature channels
            reduction (int): Reduction ratio for channel attention
        """
        super(DNAInteractionBlock, self).__init__()

        # Dual Convolutional Paths
        self.conv1_1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.conv1_2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        
        self.conv2_1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

        # Channel Attention for Interaction
        self.attention1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.attention2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # Output Projection Layer
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

        # Scaling Factor
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        """
        Forward Pass for the DNA-Inspired Dual-Path Interaction Block
        """
        # First Path (Strand 1)
        f1 = torch.relu(self.conv1_1(x))
        f1 = torch.relu(self.conv1_2(f1))

        # Second Path (Strand 2)
        f2 = torch.relu(self.conv2_1(x))
        f2 = torch.relu(self.conv2_2(f2))

        # Cross-Path Interaction
        att1 = self.attention1(f1)  # Attention on Path 1
        att2 = self.attention2(f2)  # Attention on Path 2

        i1 = att1 * f2  # Interaction from Path 2 to Path 1
        i2 = att2 * f1  # Interaction from Path 1 to Path 2

        # Feature Aggregation
        out = self.conv_out(i1 + i2)

        # Residual Connection
        out = x + self.res_scale * out

        return out

def main():
    # Define input tensor (Batch Size = 1, Channels = 64, Height = 48, Width = 48)
    x = torch.randn(1, 64, 48, 48)

    # Initialize the DNA-Inspired Block
    dna_block = DNAInteractionBlock(channels=64)

    # Forward Pass
    output = dna_block(x)

    # Print Output Shape
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {output.shape}")

if __name__ == "__main__":
    main()
