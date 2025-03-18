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

        # self.register_buffer("mean", torch.Tensor([0.46889835596084595, 0.4489615261554718, 0.40343502163887024]).view(1, 3, 1, 1))
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

# --------------------------------------------------
# LKA Module as Described in the Paper (LKA_Paper)
# --------------------------------------------------
class LKA_Paper(nn.Module):
    def __init__(self, channels):
        super(LKA_Paper, self).__init__()
        self.channels = channels
        # Depth-wise convolution: 5x5 kernel, mirror padding of 2.
        self.pad_dw = nn.ReflectionPad2d(2)
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=5, padding=0, groups=channels)
        
        # Depth-wise dilated convolution: 7x7 kernel, dilation=3, mirror padding of 9.
        self.pad_dw_dilated = nn.ReflectionPad2d(9)
        self.dw_dilated_conv = nn.Conv2d(channels, channels, kernel_size=7, dilation=3, padding=0, groups=channels)
        
        # Pointwise convolution: 1x1
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        # x: (B, C, H, W)
        # First, local context with a 5x5 depth-wise conv.
        x_dw = self.pad_dw(x)
        x_dw = self.dw_conv(x_dw)
        # Then, capture long-range dependencies with a 7x7 dilated depth-wise conv.
        x_dw_dilated = self.pad_dw_dilated(x_dw)
        x_dw_dilated = self.dw_dilated_conv(x_dw_dilated)
        # Fuse the features channel-wise via a 1x1 convolution.
        attn = self.pw_conv(x_dw_dilated)
        # Element-wise multiplication with the original input.
        return attn * x

# --------------------------------------------------
# Local Window Attention Module (same as before)
# --------------------------------------------------
class LocalWindowAttention(nn.Module):
    def __init__(self, channels, window_size=7):
        super(LocalWindowAttention, self).__init__()
        self.window_size = window_size
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        B, C, H, W = x.size()
        ws = self.window_size
        pad_h = (ws - H % ws) if H % ws != 0 else 0
        pad_w = (ws - W % ws) if W % ws != 0 else 0
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, H_pad, W_pad = x.size()
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        def window_partition(t):
            B, C, H, W = t.size()
            t = t.view(B, C, H // ws, ws, W // ws, ws)
            t = t.permute(0, 2, 4, 3, 5, 1).contiguous()
            t = t.view(B * (H // ws) * (W // ws), ws * ws, C)
            return t
        
        q_windows = window_partition(q)
        k_windows = window_partition(k)
        v_windows = window_partition(v)
        
        scale = math.sqrt(C)
        attn = torch.bmm(q_windows, k_windows.transpose(1, 2)) / scale
        attn = torch.softmax(attn, dim=-1)
        out_windows = torch.bmm(attn, v_windows)
        
        num_windows_h = H_pad // ws
        num_windows_w = W_pad // ws
        out = out_windows.view(B, num_windows_h, num_windows_w, ws, ws, C)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.view(B, C, H_pad, W_pad)
        
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
        
        out = self.out_proj(out)
        return out

class SRBlock(nn.Module):
    def __init__(self, channels, window_size=7):
        super(SRBlock, self).__init__()
        self.channels = channels
        self.window_size = window_size

        # ----------------------------
        # Stage 1: Initial Convolution and LKA Interaction
        # ----------------------------
        # Branch 1 (kept as 3x3 conv)
        self.pad3_branch1 = nn.ReflectionPad2d(1)
        self.conv1_branch1 = nn.Conv2d(channels, channels, kernel_size=3, padding=0)
        self.ln1_branch1 = nn.LayerNorm(channels)
        
        # Branch 2 (changed to 3x3 dilated conv with dilation=2)
        self.pad_dilated_branch2 = nn.ReflectionPad2d(2)  # pad=2 keeps spatial dimensions
        self.conv1_branch2 = nn.Conv2d(channels, channels, kernel_size=3, dilation=2, padding=0)
        self.ln1_branch2 = nn.LayerNorm(channels)
        
        self.prelu = nn.PReLU()
        
        # LKA Module (unchanged)
        self.lka = LKA_Paper(channels)
        self.gamma1 = nn.Parameter(torch.ones(1))
        
        # ----------------------------
        # Stage 2: Second Convolution and Local Window Attention Interaction
        # ----------------------------
        # Branch 1: 3x3 conv remains
        self.pad3_stage2_branch1 = nn.ReflectionPad2d(1)
        self.conv2_branch1 = nn.Conv2d(channels, channels, kernel_size=3, padding=0)
        self.ln2_branch1 = nn.LayerNorm(channels)
        
        # Branch 2 (changed to 3x3 dilated conv with dilation=2)
        self.pad_dilated_stage2_branch2 = nn.ReflectionPad2d(2)
        self.conv2_branch2 = nn.Conv2d(channels, channels, kernel_size=3, dilation=2, padding=0)
        self.ln2_branch2 = nn.LayerNorm(channels)
        
        # Local Window Attention Module (unchanged)
        self.lwa = LocalWindowAttention(channels, window_size)
        self.gamma2 = nn.Parameter(torch.ones(1))
        
        # Fusion Layer: merge concatenated branches
        self.conv_fusion = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.ln_fusion = nn.LayerNorm(channels)
    
    def forward(self, x):
        residual = x  # (B, C, H, W)
        B, C, H, W = x.size()
        
        # ----------------------------
        # Stage 1: Initial Convolution and LKA Interaction
        # ----------------------------
        # Branch 1: 3x3 conv
        x1 = self.pad3_branch1(x)
        x1 = self.conv1_branch1(x1)
        x1 = x1.permute(0, 2, 3, 1)
        x1 = self.ln1_branch1(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.prelu(x1)
        
        # Branch 2: 3x3 dilated conv (dilation=2)
        x2 = self.pad_dilated_branch2(x)
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
        # Branch 1: 3x3 conv
        x1_hat_padded = self.pad3_stage2_branch1(x1_hat)
        x1_2 = self.conv2_branch1(x1_hat_padded)
        x1_2 = x1_2.permute(0, 2, 3, 1)
        x1_2 = self.ln2_branch1(x1_2)
        x1_2 = x1_2.permute(0, 3, 1, 2)
        x1_2 = self.prelu(x1_2)
        
        # Branch 2: 3x3 dilated conv (dilation=2)
        x2_hat_padded = self.pad_dilated_stage2_branch2(x2_hat)
        x2_2 = self.conv2_branch2(x2_hat_padded)
        x2_2 = x2_2.permute(0, 2, 3, 1)
        x2_2 = self.ln2_branch2(x2_2)
        x2_2 = x2_2.permute(0, 3, 1, 2)
        x2_2 = self.prelu(x2_2)
        
        lwa_x2 = self.lwa(x2_2)
        lwa_x1 = self.lwa(x1_2)
        x1_2_hat = x1_2 + self.gamma2 * (lwa_x2 * x1_2)
        x2_2_hat = x2_2 + self.gamma2 * (lwa_x1 * x2_2)
        
        # ----------------------------
        # Fusion and Residual Connection
        # ----------------------------
        x_cat = torch.cat([x1_2_hat, x2_2_hat], dim=1)  # (B, 2C, H, W)
        x_fused = self.conv_fusion(x_cat)
        x_fused = x_fused.permute(0, 2, 3, 1)
        x_fused = self.ln_fusion(x_fused)
        x_fused = x_fused.permute(0, 3, 1, 2)
        x_fused = self.prelu(x_fused)
        
        out = x_fused + residual
        return out

# --------------------------------------------------
# Test Code
# --------------------------------------------------
if __name__ == "__main__":
    batch_size = 1
    channels = 64
    # Use dimensions that are not multiples of 7, e.g., 29x31.
    H, W = 29, 31
    x = torch.randn(batch_size, channels, H, W)
    
    sr_block = SRBlock(channels, window_size=7)
    output = sr_block(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
