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
            trunk.append(DNAInspireBlock(64, 64))
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

###############################################
# Module A: Dynamic Frequency Filter Generator
###############################################
class DynamicFreqFilter(nn.Module):
    def __init__(self, in_channels, hidden_dim=16):
        """
        Learns a function mapping 2D normalized coordinates to a per-channel multiplier.
        Args:
            in_channels: Number of channels for each frequency coefficient.
            hidden_dim: Hidden dimension for the MLP.
        """
        super(DynamicFreqFilter, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channels)
        )
        
    def forward(self, h, w, device):
        # Create a normalized coordinate grid in the range [-1, 1]
        y = torch.linspace(-1, 1, steps=h, device=device)
        x = torch.linspace(-1, 1, steps=w, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # shapes: (h, w)
        # Stack into a (h, w, 2) tensor where each element has (x, y)
        coords = torch.stack([grid_x, grid_y], dim=-1)  # (h, w, 2)
        # Flatten grid to (h*w, 2) and pass through MLP
        coords_flat = coords.view(-1, 2)
        weights = self.mlp(coords_flat)  # (h*w, in_channels)
        # Reshape to (1, in_channels, h, w)
        weights = weights.transpose(0, 1).view(1, -1, h, w)
        return weights

###############################################
# Module B: Pixel-wise MLP (1x1 Conv)
###############################################
class PixelMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None):
        super(PixelMLP, self).__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        return self.net(x)

###############################################
# Module C: Frequency Processor (Φ)
###############################################
class FrequencyProcessor(nn.Module):
    def __init__(self, in_channels):
        """
        Processes the concatenated low and high frequency components.
        Args:
            in_channels: Number of channels per half.
        """
        super(FrequencyProcessor, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, f_low, f_high):
        x = torch.cat([f_low, f_high], dim=1)
        out = self.conv(x)
        return self.activation(out)

###############################################
# Module D: Spatial Path
###############################################
class SpatialPath(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_bn=True, activation=nn.ReLU):
        super(SpatialPath, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(activation(inplace=True))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

###############################################
# Module E: Frequency Path (Dynamic Version)
###############################################
class FrequencyPath(nn.Module):
    def __init__(self, in_channels, phi_in_channels, out_channels, activation=nn.ReLU):
        """
        Processes the input in the frequency domain using a dynamically generated filter.
        Args:
            in_channels: Number of input channels.
            phi_in_channels: Expected channels per half for the frequency processor (should be in_channels//2).
            out_channels: Desired output channels.
        """
        super(FrequencyPath, self).__init__()
        self.dynamic_freq_filter = DynamicFreqFilter(in_channels)
        self.phi = FrequencyProcessor(phi_in_channels)
        self.activation = activation(inplace=True)
        # Project to desired output channels if necessary
        self.proj = nn.Conv2d(phi_in_channels, out_channels, kernel_size=1) \
                    if phi_in_channels != out_channels else nn.Identity()
    
    def split_frequency(self, F):
        """
        Splits F (real part) along the channel dimension into two halves.
        """
        N, C, H, W = F.shape
        c_split = C // 2
        F_low = F[:, :c_split, :, :]
        F_high = F[:, c_split:, :, :]
        return F_low, F_high
    
    def forward(self, x):
        # Compute FFT in float32 to avoid cuFFT issues with half precision.
        orig_dtype = x.dtype
        if orig_dtype == torch.float16:
            x = x.float()
        with torch.cuda.amp.autocast(enabled=False):
            F_x = torch.fft.fft2(x)
            h, w = F_x.shape[-2], F_x.shape[-1]
            weight = self.dynamic_freq_filter(h, w, x.device)  # shape: (1, in_channels, h, w)
            F_x = weight * F_x
            # For simplicity, work with the real part for splitting
            F_low, F_high = self.split_frequency(F_x.real)
            F_prime = self.phi(F_low, F_high)
            F_prime_complex = torch.complex(F_prime, torch.zeros_like(F_prime))
            x_spatial = torch.real(torch.fft.ifft2(F_prime_complex))
        x_spatial = self.proj(x_spatial)
        if orig_dtype == torch.float16:
            x_spatial = x_spatial.half()
        return x_spatial

###############################################
# Module F: Interaction (Double Bond)
###############################################
class InteractionDoubleBond(nn.Module):
    def __init__(self, feature_channels):
        """
        Fuses spatial and frequency features using a learnable scaling factor.
        """
        super(InteractionDoubleBond, self).__init__()
        self.mlp = PixelMLP(2 * feature_channels, hidden_channels=feature_channels, out_channels=feature_channels)
        self.tau = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, spatial, freq_spatial):
        lam = torch.sigmoid(self.tau)  # dynamic scaling factor λ
        concat_features = torch.cat([spatial, freq_spatial], dim=1)
        interaction = lam * self.mlp(concat_features)
        new_spatial = spatial + interaction
        new_freq = freq_spatial + interaction
        return new_spatial, new_freq, interaction, lam

###############################################
# Module G: Interaction (Triple Bond)
###############################################
class InteractionTripleBond(nn.Module):
    def __init__(self, feature_channels):
        """
        Residual fusion with an attention mechanism.
        """
        super(InteractionTripleBond, self).__init__()
        self.mlp = PixelMLP(2 * feature_channels, hidden_channels=feature_channels, out_channels=feature_channels)
        # Attention: output channels now match feature_channels for elementwise multiplication.
        self.attention = nn.Conv2d(2 * feature_channels, feature_channels, kernel_size=1)
        self.beta = nn.Parameter(torch.tensor(1.0), requires_grad=False)  # decayed per forward pass
        
    def forward(self, spatial, freq_spatial):
        concat_features = torch.cat([spatial, freq_spatial], dim=1)
        A_logits = self.attention(concat_features)
        A = F.softmax(A_logits, dim=1)
        triple_out = self.mlp(concat_features)
        beta_val = torch.clamp(self.beta * 0.99, min=0.1, max=1.0)
        interaction = beta_val * (A * triple_out)
        return interaction, beta_val

###############################################
# Module H: Final Aggregation
###############################################
class FinalAggregation(nn.Module):
    def __init__(self):
        super(FinalAggregation, self).__init__()
        
    def forward(self, I_double, I_triple, lam, beta):
        I_sum = I_double + I_triple
        eps = 1e-6
        norm = torch.norm(I_sum.view(I_sum.size(0), -1), p=2, dim=1, keepdim=True)
        norm = norm.view(-1, 1, 1, 1)
        norm = torch.clamp(norm, min=1.0)
        X_final = (lam * I_double + beta * I_triple) / (norm + eps)
        return X_final

###############################################
# Module I: DNA-Inspired Dual-Domain Block with Scaled Residual Connection
###############################################
class DNAInspireBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: Number of channels in the input.
            out_channels: Desired number of output channels.
            (Spatial dimensions are now dynamic and handled at runtime.)
            Note: Assumes in_channels is even.
        """
        super(DNAInspireBlock, self).__init__()
        # Stage 1: Parallel Feature Extraction
        self.spatial_path1 = SpatialPath(in_channels, out_channels)
        self.freq_path1 = FrequencyPath(in_channels, phi_in_channels=in_channels//2, out_channels=out_channels)
        self.interaction_double = InteractionDoubleBond(out_channels)
        
        # Stage 2: Feature Transformation
        self.spatial_path2 = SpatialPath(out_channels, out_channels)
        self.freq_path2 = FrequencyPath(out_channels, phi_in_channels=out_channels//2, out_channels=out_channels)
        self.interaction_triple = InteractionTripleBond(out_channels)
        
        self.final_aggregation = FinalAggregation()
        # Scaled residual connection from input to output:
        # If in_channels != out_channels, use a 1x1 projection to match dimensions.
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.gamma = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, X0):
        S0 = self.spatial_path1(X0)
        F_spatial0 = self.freq_path1(X0)
        S1, F1, I_double, lam = self.interaction_double(S0, F_spatial0)
        S2 = self.spatial_path2(S1)
        F_spatial2 = self.freq_path2(F1)
        I_triple, beta = self.interaction_triple(S2, F_spatial2)
        X_final = self.final_aggregation(I_double, I_triple, lam, beta)
        # Add a scaled residual connection from the input
        res = self.res_conv(X0)
        return X_final + self.gamma * res

###############################################
# Test Code
###############################################
if __name__ == '__main__':
    torch.manual_seed(42)
    
    # Define input parameters
    batch_size = 2
    in_channels = 64  # Must be even for frequency splitting
    out_channels = 64
    
    # Test with various spatial dimensions (odd and even)
    spatial_sizes = [(32, 32), (31, 31), (31, 32), (29, 35)]
    
    for (height, width) in spatial_sizes:
        X0 = torch.randn(batch_size, in_channels, height, width)
        block = DNAInspireBlock(in_channels, out_channels)
        X_final = block(X0)
        print(f"Input: ({batch_size}, {in_channels}, {height}, {width}) -> Output: {X_final.shape} | Mean: {X_final.mean().item():.4f}")

