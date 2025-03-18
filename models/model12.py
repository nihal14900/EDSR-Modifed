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
            trunk.append(ResidualConvBlock(64))
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
import math
import torch.nn.functional as F

class ConvBranch(nn.Module):
    """ One pathway with standard 3x3 Conv and another with dilated 3x3 Conv (dilation=2) """
    def __init__(self, in_channels, out_channels, use_dilation=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, 
                              padding=2 if use_dilation else 1, dilation=2 if use_dilation else 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class AdaptiveWindowedCrossAttention(nn.Module):
    """ Dynamic Windowed Cross-Attention for Super-Resolution with Padding """
    def __init__(self, dim, min_window=6, num_heads=4):
        super().__init__()
        self.min_window = min_window  # Smallest allowed window size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # Linear projections for Q, K
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)

    def get_window_size(self, H, W):
        """ Compute the dynamic window size as GCD(H, W), with a minimum threshold """
        window_size = math.gcd(H, W)  # Get greatest common divisor
        return max(window_size, self.min_window)  # Ensure it's not too small

    def pad_input(self, x, ws):
        """ Pads input to ensure divisibility by the window size """
        B, C, H, W = x.shape
        pad_h = (ws - H % ws) % ws  # Padding needed for height
        pad_w = (ws - W % ws) % ws  # Padding needed for width
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")  # Pad with reflect padding
        return x_padded, pad_h, pad_w

    def unpad_output(self, x, pad_h, pad_w):
        """ Removes extra padding from the output """
        if pad_h > 0:
            x = x[:, :, :-pad_h, :]
        if pad_w > 0:
            x = x[:, :, :, :-pad_w]
        return x

    def forward(self, x1, x2):
        B, C, H, W = x1.shape

        # Compute adaptive window size
        ws = self.get_window_size(H, W)

        # Apply padding if necessary
        x1, pad_h, pad_w = self.pad_input(x1, ws)
        x2, _, _ = self.pad_input(x2, ws)  # Same padding

        # New padded shape
        _, _, H_pad, W_pad = x1.shape

        # Reshape to non-overlapping windows
        x1_windows = x1.view(B, C, H_pad // ws, ws, W_pad // ws, ws).permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, C)
        x2_windows = x2.view(B, C, H_pad // ws, ws, W_pad // ws, ws).permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, C)

        # Compute Q, K
        Q = self.q_proj(x1_windows)  # Queries from Path A
        K = self.k_proj(x2_windows)  # Keys from Path B

        # Compute attention scores
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # [num_windows * B, ws*ws, ws*ws]
        attn = attn.softmax(dim=-1)  # Normalize attention

        # Modulate feature maps
        x1_windows = attn @ x1_windows
        x2_windows = attn @ x2_windows

        # Reshape back to original image shape
        x1_modulated = x1_windows.view(B, H_pad // ws, W_pad // ws, ws, ws, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H_pad, W_pad)
        x2_modulated = x2_windows.view(B, H_pad // ws, W_pad // ws, ws, ws, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H_pad, W_pad)

        # Remove padding
        x1_modulated = self.unpad_output(x1_modulated, pad_h, pad_w)
        x2_modulated = self.unpad_output(x2_modulated, pad_h, pad_w)

        return x1_modulated, x2_modulated
    
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, min_window=6, num_heads=4):
        super().__init__()
        self.conv3x3 = ConvBranch(in_channels, out_channels, use_dilation=False)
        self.conv3x3_dilated = ConvBranch(in_channels, out_channels, use_dilation=True)
        self.cross_attention = AdaptiveWindowedCrossAttention(dim=out_channels, min_window=min_window, num_heads=num_heads)
        self.fusion_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)
        self.res_scale = nn.Parameter(torch.ones(1) * 0.2)  # Learnable scaling

    def forward(self, x):
        identity = x
        x1 = self.conv3x3(x)
        x2 = self.conv3x3_dilated(x)
        x1_modulated, x2_modulated = self.cross_attention(x1, x2)
        out = torch.cat([x1_modulated, x2_modulated], dim=1)
        out = self.fusion_conv(out)  # Apply SE Block
        return identity + self.res_scale * out  # Residual scaling


# ===========================
#         MAIN FUNCTION
# ===========================
if __name__ == "__main__":
    # Example usage for Super-Resolution with different feature map sizes
    test_sizes = [(48, 48), (64, 64), (72, 48), (96, 80)]  # Different feature map sizes

    for H, W in test_sizes:
        B, C = 1, 64  # Batch size & channels
        x = torch.randn(B, C, H, W)

        # Instantiate the block
        cross_attention_block = CrossAttentionBlock(in_channels=C, out_channels=64, min_window=6, num_heads=4)
        
        # Forward pass
        output = cross_attention_block(x)

        print(f"Input: {H}x{W}, Window Size: {cross_attention_block.cross_attention.get_window_size(H, W)}, Output Shape: {output.shape}")
