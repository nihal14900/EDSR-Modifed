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

        self.register_buffer("mean", torch.Tensor([0.46889835596084595, 0.4489615261554718, 0.40343502163887024]).view(1, 3, 1, 1))

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
    
class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 K=4, reduction=4, temperature=30):
        """
        DynamicConv2d replaces a standard convolution with a dynamic convolution
        that aggregates K convolution kernels weighted by input-dependent attention.
        
        Args:
            in_channels (int): Number of channels in the input.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            stride (int or tuple, optional): Stride of the convolution. Default: 1.
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0.
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
            groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
            bias (bool, optional): If True, adds a learnable bias to the output. Default: True.
            K (int, optional): Number of parallel convolution kernels. Default: 4.
            reduction (int, optional): Reduction ratio for the attention branch. Default: 4.
            temperature (float, optional): Temperature parameter for softmax. Default: 30.
        """
        super(DynamicConv2d, self).__init__()
        
        # Save configuration parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.K = K
        self.temperature = temperature
        
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        
        # Define the dynamic convolution weights.
        # Shape: (K, out_channels, in_channels // groups, kH, kW)
        self.weight = nn.Parameter(
            torch.Tensor(K, out_channels, in_channels // groups, 
                         self.kernel_size[0], self.kernel_size[1])
        )
        if bias:
            # Bias for each kernel: shape (K, out_channels)
            self.bias = nn.Parameter(torch.Tensor(K, out_channels))
        else:
            self.bias = None

        # Initialize weights and bias
        for k in range(K):
            nn.init.kaiming_normal_(self.weight[k], mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        
        # Define the attention branch.
        attn_channels = max(in_channels // reduction, 1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Output shape: (B, in_channels, 1, 1)
            nn.Conv2d(in_channels, attn_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_channels, K, kernel_size=1, bias=True)
        )

    def forward(self, x):
        """
        Forward pass of dynamic convolution.
        
        Args:
            x: Input tensor of shape (B, in_channels, H, W)
        Returns:
            out: Output tensor of shape (B, out_channels, H_out, W_out)
        """
        B, C, H, W = x.shape
        
        # Compute attention scores from input.
        # Output shape: (B, K, 1, 1) -> then squeeze to (B, K)
        attn_scores = self.attention(x).view(B, self.K)
        # Normalize attention scores with softmax (using temperature)
        attn_weights = F.softmax(attn_scores / self.temperature, dim=1)  # (B, K)
        
        # Aggregate kernels using the attention weights.
        # Unsqueeze attn_weights to 6D: (B, K, 1, 1, 1, 1)
        # self.weight.unsqueeze(0) has shape: (1, K, out_channels, in_channels//groups, kH, kW)
        aggregated_weight = torch.sum(
            attn_weights.view(B, self.K, 1, 1, 1, 1) * self.weight.unsqueeze(0),
            dim=1
        )
        if self.bias is not None:
            # For bias, unsqueeze attn_weights to (B, K, 1)
            aggregated_bias = torch.sum(
                attn_weights.view(B, self.K, 1) * self.bias.unsqueeze(0),
                dim=1
            )
        else:
            aggregated_bias = None

        # Reshape input and aggregated weight for per-sample convolution:
        # 1. Reshape input from (B, C, H, W) to (1, B * C, H, W)
        x_reshaped = x.reshape(1, B * C, H, W)
        # 2. Reshape aggregated_weight from (B, out_channels, C // groups, kH, kW)
        #    to (B * out_channels, C // groups, kH, kW)
        B, Cout, Cin_per_group, kH, kW = aggregated_weight.shape
        weight_reshaped = aggregated_weight.reshape(B * Cout, Cin_per_group, kH, kW)
        
        # Reshape bias if available
        if aggregated_bias is not None:
            bias_reshaped = aggregated_bias.reshape(B * Cout)
        else:
            bias_reshaped = None
        
        # Perform convolution with groups = B * groups
        out = F.conv2d(x_reshaped, weight_reshaped, bias_reshaped,
                       stride=self.stride, padding=self.padding, 
                       dilation=self.dilation, groups=B * self.groups)
        # Reshape output back to (B, out_channels, H_out, W_out)
        out = out.reshape(B, Cout, out.shape[-2], out.shape[-1])
        return out
    
# --------------------------------------------------
# SRBlock: Super Resolution Block with Two Parallel Paths using Dynamic Convolutions
# --------------------------------------------------
class SRBlock(nn.Module):
    def __init__(self, channels, window_size=7):
        super(SRBlock, self).__init__()
        self.channels = channels
        self.window_size = window_size

        # Stage 1: Initial 3x3 Dynamic Convolutions (with mirror padding) for two parallel branches.
        self.pad3 = nn.ReflectionPad2d(1)
        self.conv1_branch1 = DynamicConv2d(channels, channels, kernel_size=3, padding=0, K=4, reduction=4, temperature=30)
        self.conv1_branch2 = DynamicConv2d(channels, channels, kernel_size=3, padding=0, K=4, reduction=4, temperature=30)
        
        self.ln1_branch1 = nn.LayerNorm(channels)
        self.ln1_branch2 = nn.LayerNorm(channels)
        self.prelu = nn.PReLU()
        
        # Stage 1: LKA Module (from the paper)
        self.lka = LKA_Paper(channels)
        self.gamma1 = nn.Parameter(torch.ones(1))
        
        # Stage 2: Second 3x3 Dynamic Convolutions for each branch.
        self.pad3_stage2 = nn.ReflectionPad2d(1)
        self.conv2_branch1 = DynamicConv2d(channels, channels, kernel_size=3, padding=0, K=4, reduction=4, temperature=30)
        self.conv2_branch2 = DynamicConv2d(channels, channels, kernel_size=3, padding=0, K=4, reduction=4, temperature=30)
        self.ln2_branch1 = nn.LayerNorm(channels)
        self.ln2_branch2 = nn.LayerNorm(channels)
        
        # Stage 2: Local Window Attention Module.
        self.lwa = LocalWindowAttention(channels, window_size)
        self.gamma2 = nn.Parameter(torch.ones(1))
        
        # Fusion Layer: 1x1 dynamic convolution to merge concatenated branches.
        self.conv_fusion = DynamicConv2d(channels * 2, channels, kernel_size=1, padding=0, K=4, reduction=4, temperature=30)
        self.ln_fusion = nn.LayerNorm(channels)
    
    def forward(self, x):
        residual = x  # (B, C, H, W)
        B, C, H, W = x.size()
        
        # ----------------------------
        # Stage 1: Initial Convolution and LKA Interaction
        # ----------------------------
        x1 = self.pad3(x)
        x1 = self.conv1_branch1(x1)
        x1 = x1.permute(0, 2, 3, 1)
        x1 = self.ln1_branch1(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.prelu(x1)
        
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
        x1_hat_padded = self.pad3_stage2(x1_hat)
        x1_2 = self.conv2_branch1(x1_hat_padded)
        x1_2 = x1_2.permute(0, 2, 3, 1)
        x1_2 = self.ln2_branch1(x1_2)
        x1_2 = x1_2.permute(0, 3, 1, 2)
        x1_2 = self.prelu(x1_2)
        
        x2_hat_padded = self.pad3_stage2(x2_hat)
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
