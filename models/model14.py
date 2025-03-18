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
            trunk.append(SRBlock(64, window_size=7))
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
import math

# --------------------------------------------
# Large Kernel Attention (LKA) Module
# --------------------------------------------
class LKA(nn.Module):
    def __init__(self, channels):
        super(LKA, self).__init__()
        # For a 7x7 convolution, use ReflectionPad2d with pad=3 on all sides.
        self.pad = nn.ReflectionPad2d(3)
        self.conv = nn.Conv2d(channels, channels, kernel_size=7, padding=0)
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.pad(x)
        return self.conv(x)

# --------------------------------------------
# Local Window Attention (LA) Module
# --------------------------------------------
class LocalWindowAttention(nn.Module):
    def __init__(self, channels, window_size=7):
        super(LocalWindowAttention, self).__init__()
        self.window_size = window_size
        # Using conv1x1 layers as linear projections for Q, K, V.
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.size()
        ws = self.window_size

        # If H or W is not a multiple of ws, pad (using mirror/reflective padding)
        pad_h = (ws - H % ws) if H % ws != 0 else 0
        pad_w = (ws - W % ws) if W % ws != 0 else 0
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, H_pad, W_pad = x.size()
        
        # Compute Q, K, V on the padded input.
        q = self.q_proj(x)  # (B, C, H_pad, W_pad)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Partition feature maps into non-overlapping windows.
        # Helper: reshape tensor into windows of size (ws x ws).
        def window_partition(t):
            # t: (B, C, H_pad, W_pad)
            B, C, H, W = t.size()
            t = t.view(B, C, H // ws, ws, W // ws, ws)
            t = t.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, H//ws, W//ws, ws, ws, C)
            t = t.view(B * (H // ws) * (W // ws), ws * ws, C)  # (B*num_windows, ws*ws, C)
            return t
        
        q_windows = window_partition(q)
        k_windows = window_partition(k)
        v_windows = window_partition(v)
        
        # Compute scaled dot-product attention within each window.
        # q_windows, k_windows, v_windows: (B*num_windows, ws*ws, C)
        scale = math.sqrt(C)
        attn = torch.bmm(q_windows, k_windows.transpose(1, 2)) / scale  # (B*num_windows, ws*ws, ws*ws)
        attn = torch.softmax(attn, dim=-1)
        out_windows = torch.bmm(attn, v_windows)  # (B*num_windows, ws*ws, C)
        
        # Merge windows back to feature map shape.
        num_windows_h = H_pad // ws
        num_windows_w = W_pad // ws
        out = out_windows.view(B, num_windows_h, num_windows_w, ws, ws, C)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B, C, num_windows_h, ws, num_windows_w, ws)
        out = out.view(B, C, H_pad, W_pad)
        
        # Remove any extra padding if it was added.
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
        
        out = self.out_proj(out)
        return out

# --------------------------------------------
# Super Resolution Block with Two Parallel Paths
# --------------------------------------------
class SRBlock(nn.Module):
    def __init__(self, channels, window_size=7):
        super(SRBlock, self).__init__()
        self.channels = channels
        self.window_size = window_size

        # Stage 1: Initial 3x3 Convolutions (with mirror padding) for two parallel branches.
        # Using ReflectionPad2d for mirror padding.
        self.pad3 = nn.ReflectionPad2d(1)
        self.conv1_branch1 = nn.Conv2d(channels, channels, kernel_size=3, padding=0)
        self.conv1_branch2 = nn.Conv2d(channels, channels, kernel_size=3, padding=0)
        
        # LayerNorm applied per pixel over the channel dimension.
        self.ln1_branch1 = nn.LayerNorm(channels)
        self.ln1_branch2 = nn.LayerNorm(channels)
        self.prelu = nn.PReLU()
        
        # Stage 1: LKA Module (shared for simplicity)
        self.lka = LKA(channels)
        self.gamma1 = nn.Parameter(torch.ones(1))
        
        # Stage 2: Second 3x3 Convolutions for each branch.
        self.pad3_stage2 = nn.ReflectionPad2d(1)
        self.conv2_branch1 = nn.Conv2d(channels, channels, kernel_size=3, padding=0)
        self.conv2_branch2 = nn.Conv2d(channels, channels, kernel_size=3, padding=0)
        self.ln2_branch1 = nn.LayerNorm(channels)
        self.ln2_branch2 = nn.LayerNorm(channels)
        
        # Stage 2: Local Window Attention Module (shared for simplicity)
        self.lwa = LocalWindowAttention(channels, window_size)
        self.gamma2 = nn.Parameter(torch.ones(1))
        
        # Fusion Layer: 1x1 convolution to merge the concatenated branches.
        self.conv_fusion = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.ln_fusion = nn.LayerNorm(channels)
    
    def forward(self, x):
        # x: (B, C, H, W)
        residual = x
        B, C, H, W = x.size()
        
        # ----------------------------
        # Stage 1: Initial Convolution and LKA Interaction
        # ----------------------------
        # Process branch 1.
        x1 = self.pad3(x)
        x1 = self.conv1_branch1(x1)  # (B, C, H, W)
        # Permute to (B, H, W, C) for LayerNorm over channels.
        x1 = x1.permute(0, 2, 3, 1)
        x1 = self.ln1_branch1(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.prelu(x1)
        
        # Process branch 2.
        x2 = self.pad3(x)
        x2 = self.conv1_branch2(x2)
        x2 = x2.permute(0, 2, 3, 1)
        x2 = self.ln1_branch2(x2)
        x2 = x2.permute(0, 3, 1, 2)
        x2 = self.prelu(x2)
        
        # Cross-branch modulation using LKA.
        lka_x2 = self.lka(x2)
        lka_x1 = self.lka(x1)
        x1_hat = x1 + self.gamma1 * (lka_x2 * x1)
        x2_hat = x2 + self.gamma1 * (lka_x1 * x2)
        
        # ----------------------------
        # Stage 2: Second Convolution and Local Window Attention Interaction
        # ----------------------------
        # Branch 1 second convolution.
        x1_hat_padded = self.pad3_stage2(x1_hat)
        x1_2 = self.conv2_branch1(x1_hat_padded)
        x1_2 = x1_2.permute(0, 2, 3, 1)
        x1_2 = self.ln2_branch1(x1_2)
        x1_2 = x1_2.permute(0, 3, 1, 2)
        x1_2 = self.prelu(x1_2)
        
        # Branch 2 second convolution.
        x2_hat_padded = self.pad3_stage2(x2_hat)
        x2_2 = self.conv2_branch2(x2_hat_padded)
        x2_2 = x2_2.permute(0, 2, 3, 1)
        x2_2 = self.ln2_branch2(x2_2)
        x2_2 = x2_2.permute(0, 3, 1, 2)
        x2_2 = self.prelu(x2_2)
        
        # Cross-branch modulation using Local Window Attention.
        lwa_x2 = self.lwa(x2_2)
        lwa_x1 = self.lwa(x1_2)
        x1_2_hat = x1_2 + self.gamma2 * (lwa_x2 * x1_2)
        x2_2_hat = x2_2 + self.gamma2 * (lwa_x1 * x2_2)
        
        # ----------------------------
        # Fusion and Residual Connection
        # ----------------------------
        # Concatenate along the channel dimension.
        x_cat = torch.cat([x1_2_hat, x2_2_hat], dim=1)  # (B, 2C, H, W)
        x_fused = self.conv_fusion(x_cat)
        x_fused = x_fused.permute(0, 2, 3, 1)
        x_fused = self.ln_fusion(x_fused)
        x_fused = x_fused.permute(0, 3, 1, 2)
        x_fused = self.prelu(x_fused)
        
        # Residual connection.
        out = x_fused + residual
        return out

# --------------------------------------------
# Test Code
# --------------------------------------------
if __name__ == "__main__":
    # Define hyperparameters.
    batch_size = 1
    channels = 64
    # Choose height and width that are NOT multiples of 7 (e.g., 29x31).
    H, W = 29, 31
    
    # Create a random input tensor.
    x = torch.randn(batch_size, channels, H, W)
    
    # Instantiate the block.
    sr_block = SRBlock(channels, window_size=7)
    
    # Forward pass.
    output = sr_block(x)
    
    # Print the shapes to verify.
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
