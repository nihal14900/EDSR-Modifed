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
            trunk.append(DoubleTripleDNABlock(channels=64, kernel_size=3,
                                 dbl_num_interactions=2,
                                 trp_num_interactions=3))
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

class DoubleTripleDNABlock(nn.Module):
    """
    A block that first applies an initial convolution,
    then a 'double bond' (2 cross-interactions),
    then an intermediate convolution,
    then a 'triple bond' (3 cross-interactions),
    and finally adds a residual skip to the original input.
    """
    def __init__(self, channels, kernel_size=3, 
                 dbl_num_interactions=2,  # Typically 2
                 trp_num_interactions=3): # Typically 3
        super(DoubleTripleDNABlock, self).__init__()
        
        # -- Initial convolution
        self.init_conv = nn.Conv2d(channels, channels, kernel_size, 
                                   padding=kernel_size // 2)
        
        # -- Double bond streams (two strands)
        self.extract1_dbl = nn.Conv2d(channels, channels, kernel_size,
                                      padding=kernel_size // 2)
        self.extract2_dbl = nn.Conv2d(channels, channels, kernel_size,
                                      padding=kernel_size // 2)
        
        # -- Double bond cross-interactions
        # (Shared weights across dbl_num_interactions)
        self.U12_dbl = nn.Conv2d(channels, channels, kernel_size, 
                                 padding=kernel_size // 2)
        self.U21_dbl = nn.Conv2d(channels, channels, kernel_size, 
                                 padding=kernel_size // 2)
        
        # -- Intermediate convolution
        self.mid_conv = nn.Conv2d(channels, channels, kernel_size,
                                  padding=kernel_size // 2)
        
        # -- Triple bond streams (two strands)
        self.extract1_trp = nn.Conv2d(channels, channels, kernel_size,
                                      padding=kernel_size // 2)
        self.extract2_trp = nn.Conv2d(channels, channels, kernel_size,
                                      padding=kernel_size // 2)
        
        # -- Triple bond cross-interactions
        # (Shared weights across trp_num_interactions)
        self.U12_trp = nn.Conv2d(channels, channels, kernel_size, 
                                 padding=kernel_size // 2)
        self.U21_trp = nn.Conv2d(channels, channels, kernel_size, 
                                 padding=kernel_size // 2)
        
        # -- Activation function
        self.activation = nn.ReLU(inplace=True)
        
        # -- Interaction counts
        self.dbl_num_interactions = dbl_num_interactions
        self.trp_num_interactions = trp_num_interactions

    def forward(self, x):
        # 1) Initial Convolution
        z0 = self.activation(self.init_conv(x))
        
        # 2) Double Bond ------------------------------------
        #    Create two parallel feature streams
        z1_dbl = self.activation(self.extract1_dbl(z0))
        z2_dbl = self.activation(self.extract2_dbl(z0))
        
        #    Two cross-interaction steps
        for _ in range(self.dbl_num_interactions):
            g12 = torch.sigmoid(self.U12_dbl(z2_dbl))
            g21 = torch.sigmoid(self.U21_dbl(z1_dbl))
            
            z1_dbl = z1_dbl * g12
            z2_dbl = z2_dbl * g21
        
        #    Merge
        M_dbl = z1_dbl + z2_dbl
        
        # 3) Intermediate Convolution
        z1 = self.activation(self.mid_conv(M_dbl))
        
        # 4) Triple Bond ------------------------------------
        #    Create two parallel feature streams
        z1_trp = self.activation(self.extract1_trp(z1))
        z2_trp = self.activation(self.extract2_trp(z1))
        
        #    Three cross-interaction steps
        for _ in range(self.trp_num_interactions):
            g12 = torch.sigmoid(self.U12_trp(z2_trp))
            g21 = torch.sigmoid(self.U21_trp(z1_trp))
            
            z1_trp = z1_trp * g12
            z2_trp = z2_trp * g21
        
        #    Merge
        M_trp = z1_trp + z2_trp
        
        # 5) Residual skip connection
        return x + M_trp

# --------------------------------------------------------------------
# Example usage & test code:
if __name__ == "__main__":
    # Example: 4 images, 64 feature channels, 32x32 resolution
    dummy_input = torch.randn(4, 64, 32, 32)
    
    # Create the Double-Triple DNA block
    block = DoubleTripleDNABlock(channels=64, kernel_size=3,
                                 dbl_num_interactions=2,
                                 trp_num_interactions=3)
    
    # Forward pass
    output = block(dummy_input)
    
    print("Input shape: ", dummy_input.shape)
    print("Output shape:", output.shape)
