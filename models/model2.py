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
            trunk.append(DNABlock(channels=64, kernel_size=3, num_interactions=2))
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

class DNABlock(nn.Module):
    """
    DNA-inspired block with two parallel feature streams and iterative cross-interaction.
    """
    def __init__(self, channels, kernel_size=3, num_interactions=2):
        """
        Args:
            channels (int): Number of input and output channels.
            kernel_size (int): Convolution kernel size for feature transforms.
            num_interactions (int): How many times to repeat cross-interaction.
        """
        super(DNABlock, self).__init__()
        
        # Parallel extraction layers (two "strands")
        self.extract1 = nn.Conv2d(channels, channels, kernel_size, 
                                  padding=kernel_size // 2)
        self.extract2 = nn.Conv2d(channels, channels, kernel_size, 
                                  padding=kernel_size // 2)
        
        # Cross-interaction (base-pair) layers
        self.w12 = nn.Conv2d(channels, channels, kernel_size, 
                             padding=kernel_size // 2)
        self.w21 = nn.Conv2d(channels, channels, kernel_size, 
                             padding=kernel_size // 2)
        
        # ReLU activation (can be replaced with e.g. LeakyReLU)
        self.activation = nn.ReLU(inplace=True)
        
        # Number of times to repeat cross-interaction
        self.num_interactions = num_interactions

    def forward(self, x):
        # Parallel feature extraction
        z1 = self.activation(self.extract1(x))
        z2 = self.activation(self.extract2(x))

        # Iterative cross-interactions
        for _ in range(self.num_interactions):
            g12 = torch.sigmoid(self.w12(z2))  # Gate for z1 based on z2
            g21 = torch.sigmoid(self.w21(z1))  # Gate for z2 based on z1
            
            z1 = z1 * g12
            z2 = z2 * g21

        # Merge two strands
        merged = z1 + z2
        
        # Residual connection
        return x + merged

# Example usage:
if __name__ == "__main__":
    # Suppose we have a batch of 4 images, 64 feature channels, 32x32 resolution
    dummy_input = torch.randn(4, 64, 32, 32)
    
    # Create a DNA block
    dna_block = DNABlock(channels=64, kernel_size=3, num_interactions=2)
    
    # Forward pass
    output = dna_block(dummy_input)
    print("Output shape:", output.shape)
