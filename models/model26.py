import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class TransformerChannelAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.LayerNorm([channels])
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm(x_flat)
        x_norm = x_norm.transpose(1, 2).reshape(B, C, H, W)
        qkv = self.qkv(x_norm).reshape(B, 3, self.num_heads, self.head_dim, H*W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x_attn = (attn @ v).reshape(B, C, H, W)
        x_attn = self.proj(x_attn)
        attn_map = self.sigmoid(x_attn.mean(dim=(2, 3), keepdim=True))
        return attn_map

class StandardSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return self.sigmoid(y)

class TransformerSpatialAttention(nn.Module):
    def __init__(self, channels, window_size=8, num_heads=8):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        B, C, H, W = x.shape
        num_h_windows = (H + self.window_size - 1) // self.window_size
        num_w_windows = (W + self.window_size - 1) // self.window_size
        H_pad = num_h_windows * self.window_size
        W_pad = num_w_windows * self.window_size
        padding_h = H_pad - H
        padding_w = W_pad - W
        if padding_h > 0 or padding_w > 0:
            x = F.pad(x, (0, padding_w, 0, padding_h))
        x_windows = x.reshape(B, C, num_h_windows, self.window_size, num_w_windows, self.window_size)
        x_windows = x_windows.permute(0, 2, 4, 3, 5, 1)
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, C)
        x_norm = self.norm(x_windows)
        qkv = self.qkv(x_norm).reshape(-1, self.window_size * self.window_size, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x_attn = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        x_attn = self.proj(x_attn)
        x_attn = x_attn.reshape(B, num_h_windows, num_w_windows, self.window_size, self.window_size, C)
        x_attn = x_attn.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H_pad, W_pad)
        if padding_h > 0 or padding_w > 0:
            x_attn = x_attn[:, :, :H, :W]
        attn_map = self.sigmoid(x_attn.mean(dim=1, keepdim=True))
        return attn_map

class SpatialFrequencyAttentionBlock(nn.Module):
    def __init__(self, channels, reduction=16, window_size=8, num_heads=8):
        super().__init__()
        
        # Spatial domain pathway
        self.spatial_stage1_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.spatial_stage1_norm = nn.BatchNorm2d(channels)
        self.spatial_stage1_act = nn.PReLU()
        
        self.spatial_stage2_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.spatial_stage2_norm = nn.BatchNorm2d(channels)
        self.spatial_stage2_act = nn.PReLU()
        
        self.spatial_stage3_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.spatial_stage3_norm = nn.BatchNorm2d(channels)
        self.spatial_stage3_act = nn.PReLU()
        
        self.spatial_stage4_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.spatial_stage4_norm = nn.BatchNorm2d(channels)
        self.spatial_stage4_act = nn.PReLU()
        
        # Dilated convolutional pathway
        self.dilated_stage1_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.dilated_stage1_norm = nn.BatchNorm2d(channels)
        self.dilated_stage1_act = nn.PReLU()
        
        self.dilated_stage2_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.dilated_stage2_norm = nn.BatchNorm2d(channels)
        self.dilated_stage2_act = nn.PReLU()
        
        self.dilated_stage3_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.dilated_stage3_norm = nn.BatchNorm2d(channels)
        self.dilated_stage3_act = nn.PReLU()
        
        self.dilated_stage4_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.dilated_stage4_norm = nn.BatchNorm2d(channels)
        self.dilated_stage4_act = nn.PReLU()
        
        # Attention modules
        self.std_channel_attn = StandardChannelAttention(channels, reduction)
        self.transformer_channel_attn = TransformerChannelAttention(channels, num_heads)
        self.std_spatial_attn = StandardSpatialAttention()
        self.transformer_spatial_attn = TransformerSpatialAttention(channels, window_size, num_heads)
        
        # Feature fusion
        self.fusion_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.fusion_norm = nn.BatchNorm2d(channels)
        self.fusion_act = nn.PReLU()
        
        # Final normalization
        self.final_norm = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        identity = x
        
        # Stage 1: Compute attention scores before applying
        spatial_stage1 = self.spatial_stage1_conv(x)
        spatial_stage1 = self.spatial_stage1_norm(spatial_stage1)
        spatial_stage1 = self.spatial_stage1_act(spatial_stage1)
        
        dilated_stage1 = self.dilated_stage1_conv(x)
        dilated_stage1 = self.dilated_stage1_norm(dilated_stage1)
        dilated_stage1 = self.dilated_stage1_act(dilated_stage1)
        
        # Compute attention scores
        spatial_attn1 = self.std_channel_attn(spatial_stage1)
        dilated_attn1 = self.std_channel_attn(dilated_stage1)
        
        # Apply attention scores
        spatial_stage1 = spatial_stage1 * dilated_attn1  # Dilated modulates spatial
        dilated_stage1 = dilated_stage1 * spatial_attn1  # Spatial modulates dilated
        
        # Stage 2: Transformer channel attention
        spatial_stage2 = self.spatial_stage2_conv(spatial_stage1)
        spatial_stage2 = self.spatial_stage2_norm(spatial_stage2)
        spatial_stage2 = self.spatial_stage2_act(spatial_stage2)
        
        dilated_stage2 = self.dilated_stage2_conv(dilated_stage1)
        dilated_stage2 = self.dilated_stage2_norm(dilated_stage2)
        dilated_stage2 = self.dilated_stage2_act(dilated_stage2)
        
        # Compute attention scores
        spatial_attn2 = self.transformer_channel_attn(spatial_stage2)
        dilated_attn2 = self.transformer_channel_attn(dilated_stage2)
        
        # Apply attention scores
        spatial_stage2 = spatial_stage2 * dilated_attn2  # Dilated modulates spatial
        dilated_stage2 = dilated_stage2 * spatial_attn2  # Spatial modulates dilated
        
        # Stage 3: Standard spatial attention
        spatial_stage3 = self.spatial_stage3_conv(spatial_stage2)
        spatial_stage3 = self.spatial_stage3_norm(spatial_stage3)
        spatial_stage3 = self.spatial_stage3_act(spatial_stage3)
        
        dilated_stage3 = self.dilated_stage3_conv(dilated_stage2)
        dilated_stage3 = self.dilated_stage3_norm(dilated_stage3)
        dilated_stage3 = self.dilated_stage3_act(dilated_stage3)
        
        # Compute attention scores
        spatial_attn3 = self.std_spatial_attn(spatial_stage3)
        dilated_attn3 = self.std_spatial_attn(dilated_stage3)
        
        # Apply attention scores
        spatial_stage3 = spatial_stage3 * dilated_attn3  # Dilated modulates spatial
        dilated_stage3 = dilated_stage3 * spatial_attn3  # Spatial modulates dilated
        
        # Stage 4: Transformer spatial attention
        spatial_stage4 = self.spatial_stage4_conv(spatial_stage3)
        spatial_stage4 = self.spatial_stage4_norm(spatial_stage4)
        spatial_stage4 = self.spatial_stage4_act(spatial_stage4)
        
        dilated_stage4 = self.dilated_stage4_conv(dilated_stage3)
        dilated_stage4 = self.dilated_stage4_norm(dilated_stage4)
        dilated_stage4 = self.dilated_stage4_act(dilated_stage4)
        
        # Compute attention scores
        spatial_attn4 = self.transformer_spatial_attn(spatial_stage4)
        dilated_attn4 = self.transformer_spatial_attn(dilated_stage4)
        
        # Apply attention scores
        spatial_stage4 = spatial_stage4 * dilated_attn4  # Dilated modulates spatial
        dilated_stage4 = dilated_stage4 * spatial_attn4  # Spatial modulates dilated
        
        # Fusion of spatial and dilated paths
        concat_features = torch.cat([spatial_stage4, dilated_stage4], dim=1)
        fused_features = self.fusion_conv(concat_features)
        fused_features = self.fusion_norm(fused_features)
        fused_features = self.fusion_act(fused_features)
        
        # Residual connection
        output = self.final_norm(fused_features + identity)
        
        return output

def test_model():
    model = SpatialFrequencyAttentionBlock(64)
    
    x = torch.randn(2, 64, 32, 32)
    out = model(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    
    x = torch.randn(2, 64, 33, 33)
    out = model(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    
    x = torch.randn(2, 64, 37, 41)
    out = model(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    
    if torch.cuda.is_available():
        model = model.cuda().half()
        x = torch.randn(2, 64, 48, 48).cuda().half()
        out = model(x)
        print(f"Half precision input shape: {x.shape}, Output shape: {out.shape}")

if __name__ == "__main__":
    test_model()

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
            trunk.append(SpatialFrequencyAttentionBlock(64))
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
