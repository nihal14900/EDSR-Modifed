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
            trunk.append(SRBlockDualDomain(channels=64, window_size=7))
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


#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ----------------------------------------------------
# 1) DCT/IDCT Helpers
# ----------------------------------------------------
def dct_1d(x):
    """
    Naive DCT-II along the last dimension of x.
    x: (..., N)
    returns: (..., N)
    """
    N = x.shape[-1]
    device = x.device

    # Create transform matrix for DCT-II
    k = torch.arange(N, device=device).reshape(-1, 1)  # shape (N,1)
    n = torch.arange(N, device=device).reshape(1, -1)  # shape (1,N)
    dct_mat = torch.cos(math.pi * (2*n + 1) * k / (2.0*N)) * math.sqrt(2.0/N)

    # Scale for DC term
    dct_mat[0, :] = dct_mat[0, :] / math.sqrt(2.0)

    # (..., N) x (N, N) => (..., N)
    X = torch.matmul(x, dct_mat.transpose(0, 1))
    return X

def idct_1d(X):
    """
    Naive Inverse DCT-II along the last dimension of X.
    X: (..., N)
    returns: (..., N)
    """
    N = X.shape[-1]
    device = X.device

    # Create inverse DCT matrix
    k = torch.arange(N, device=device).reshape(-1, 1)
    n = torch.arange(N, device=device).reshape(1, -1)
    idct_mat = torch.cos(math.pi * (2*k + 1) * n / (2.0*N)) * math.sqrt(2.0/N)
    idct_mat[:, 0] = idct_mat[:, 0] / math.sqrt(2.0)

    # (..., N) x (N, N) => (..., N)
    x = torch.matmul(X, idct_mat.transpose(0, 1))
    return x

def dct_2d(x):
    """
    2D DCT-II: apply dct_1d along W, then along H.
    x: (B, C, H, W)
    returns: (B, C, H, W)
    """
    B, C, H, W = x.size()
    # Flatten (B*C, H, W)
    # x = x.view(B*C, H, W)
    x = x.reshape(B*C, H, W)

    # 1D DCT along width
    x = dct_1d(x)  # shape: (B*C, H, W)

    # Transpose so we can apply DCT along height
    x = x.transpose(1, 2)  # (B*C, W, H)
    x = dct_1d(x)          # (B*C, W, H)
    x = x.transpose(1, 2)  # (B*C, H, W)

    # Reshape back
    x = x.reshape(B, C, H, W)
    return x

def idct_2d(X):
    """
    2D inverse DCT-II: apply idct_1d along W, then along H.
    X: (B, C, H, W)
    returns: (B, C, H, W)
    """
    B, C, H, W = X.size()
    X = X.reshape(B*C, H, W)

    # 1D IDCT along width
    X = idct_1d(X)

    # Transpose so we can apply IDCT along height
    X = X.transpose(1, 2)
    X = idct_1d(X)
    X = X.transpose(1, 2)

    X = X.reshape(B, C, H, W)
    return X

# ----------------------------------------------------
# 2) LKA (Large Kernel Attention) Module
#    (As in your original code)
# ----------------------------------------------------
class LKA_Paper(nn.Module):
    def __init__(self, channels):
        super(LKA_Paper, self).__init__()
        self.channels = channels
        # Depth-wise convolution: 5x5 kernel, reflection pad of 2
        self.pad_dw = nn.ReflectionPad2d(2)
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=5, padding=0, groups=channels)
        
        # Depth-wise dilated convolution: 7x7 kernel, dilation=3, reflection pad of 9
        self.pad_dw_dilated = nn.ReflectionPad2d(9)
        self.dw_dilated_conv = nn.Conv2d(channels, channels, kernel_size=7, dilation=3, padding=0, groups=channels)
        
        # Pointwise convolution: 1x1
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        # x: (B, C, H, W)
        # 1) 5x5 depthwise conv
        x_dw = self.pad_dw(x)
        x_dw = self.dw_conv(x_dw)
        # 2) 7x7 dilated depthwise conv
        x_dw_dilated = self.pad_dw_dilated(x_dw)
        x_dw_dilated = self.dw_dilated_conv(x_dw_dilated)
        # 3) fuse via 1x1 conv
        attn = self.pw_conv(x_dw_dilated)
        return attn * x

# ----------------------------------------------------
# 3) Local Window Attention
#    (As in your original code)
# ----------------------------------------------------
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
        
        # Reflection pad if needed
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, H_pad, W_pad = x.size()
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Partition into windows
        def window_partition(t):
            B_, C_, H_, W_ = t.size()
            t = t.view(B_, C_, H_ // ws, ws, W_ // ws, ws)
            t = t.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B_*#win, ws*ws, C_)
            t = t.view(B_ * (H_ // ws) * (W_ // ws), ws * ws, C_)
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

# ----------------------------------------------------
# 4) SRBlockDualDomain
#    One branch in spatial domain, the other in DCT domain.
# ----------------------------------------------------
class SRBlockDualDomain(nn.Module):
    def __init__(self, channels, window_size=7):
        super().__init__()
        self.channels = channels
        
        # ----- Spatial Branch -----
        self.conv1_branch1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ln1_branch1 = nn.LayerNorm(channels)
        self.prelu = nn.PReLU()  # Shared PReLU for simplicity
        
        self.lka_spatial = LKA_Paper(channels)
        
        self.conv2_branch1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ln2_branch1 = nn.LayerNorm(channels)
        self.lwa_spatial = LocalWindowAttention(channels, window_size)
        
        # ----- Frequency Branch -----
        self.conv1_branch2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ln1_branch2 = nn.LayerNorm(channels)
        self.lka_freq = LKA_Paper(channels)
        
        self.conv2_branch2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ln2_branch2 = nn.LayerNorm(channels)
        self.lwa_freq = LocalWindowAttention(channels, window_size)
        
        # Cross-Branch Gains
        self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma2 = nn.Parameter(torch.ones(1))
        
        # Fusion
        self.conv_fusion = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.ln_fusion = nn.LayerNorm(channels)
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        residual = x
        
        # ------------------------------------------------
        # Branch 1: SPATIAL domain
        # ------------------------------------------------
        x1 = self.conv1_branch1(x)
        x1 = x1.permute(0, 2, 3, 1)
        x1 = self.ln1_branch1(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.prelu(x1)
        
        lka_x1 = self.lka_spatial(x1)
        
        # ------------------------------------------------
        # Branch 2: FREQUENCY domain
        # ------------------------------------------------
        x_freq = dct_2d(x)  # transform input to freq domain
        x2 = self.conv1_branch2(x_freq)
        x2 = x2.permute(0, 2, 3, 1)
        x2 = self.ln1_branch2(x2)
        x2 = x2.permute(0, 3, 1, 2)
        x2 = self.prelu(x2)
        
        lka_x2 = self.lka_freq(x2)
        
        # Convert freq-domain output back to spatial for cross-branch
        lka_x2_spatial = idct_2d(lka_x2)
        
        # Cross-branch #1
        x1_hat = x1 + self.gamma1 * (lka_x2_spatial * x1)
        
        # For symmetrical cross, we transform lka_x1 -> freq
        lka_x1_freq = dct_2d(lka_x1)
        x2_hat = x2 + self.gamma1 * (lka_x1_freq * x2)
        
        # 2nd stage for SPATIAL
        x1_2 = self.conv2_branch1(x1_hat)
        x1_2 = x1_2.permute(0, 2, 3, 1)
        x1_2 = self.ln2_branch1(x1_2)
        x1_2 = x1_2.permute(0, 3, 1, 2)
        x1_2 = self.prelu(x1_2)
        
        lwa_x1 = self.lwa_spatial(x1_2)
        
        # 2nd stage for FREQUENCY
        x2_hat_freq = dct_2d(x2_hat)
        x2_2 = self.conv2_branch2(x2_hat_freq)
        x2_2 = x2_2.permute(0, 2, 3, 1)
        x2_2 = self.ln2_branch2(x2_2)
        x2_2 = x2_2.permute(0, 3, 1, 2)
        x2_2 = self.prelu(x2_2)
        
        lwa_x2_freq = self.lwa_freq(x2_2)
        lwa_x2_spatial = idct_2d(lwa_x2_freq)
        
        # Cross-branch #2
        x1_2_hat = x1_2 + self.gamma2 * (lwa_x2_spatial * x1_2)
        
        # symmetrical cross again
        lwa_x1_freq = dct_2d(lwa_x1)
        x2_2_hat = x2_2 + self.gamma2 * (lwa_x1_freq * x2_2)
        
        # convert freq back to spatial for final fusion
        x2_2_hat_spatial = idct_2d(x2_2_hat)
        
        # Fuse
        x_cat = torch.cat([x1_2_hat, x2_2_hat_spatial], dim=1)  # (B, 2C, H, W)
        x_fused = self.conv_fusion(x_cat)
        x_fused = x_fused.permute(0, 2, 3, 1)
        x_fused = self.ln_fusion(x_fused)
        x_fused = x_fused.permute(0, 3, 1, 2)
        x_fused = self.prelu(x_fused)
        
        # Residual
        out = x_fused + residual
        return out

# ----------------------------------------------------
# 5) Test / Demo
# ----------------------------------------------------
def main():
    # Create a random input tensor
    B, C, H, W = 1, 64, 32, 32
    x = torch.randn(B, C, H, W)

    # Instantiate the dual-domain SR block
    block = SRBlockDualDomain(channels=C, window_size=7)

    # Forward pass
    out = block(x)
    print("Input shape: ", x.shape)
    print("Output shape:", out.shape)

if __name__ == "__main__":
    main()
