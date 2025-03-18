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
import torch.nn.functional as F

def window_partition(x, window_size):
    """
    Partition the input feature map into non-overlapping windows.

    Args:
        x (torch.Tensor): Tensor of shape (B, H, W, C)
        window_size (int): Window size.
    
    Returns:
        windows (torch.Tensor): Tensor of shape (num_windows*B, window_size*window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Reverse the window partitioning to reconstruct the feature map.

    Args:
        windows (torch.Tensor): Tensor of shape (num_windows*B, window_size*window_size, C)
        window_size (int): Window size.
        H (int): Height of the padded feature map.
        W (int): Width of the padded feature map.
    
    Returns:
        x (torch.Tensor): Reconstructed tensor of shape (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Mlp(nn.Module):
    """
    Simple two-layer MLP with GELU activation and dropout.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with relative positional bias.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        """
        Args:
            dim (int): Number of input channels.
            window_size (int): Fixed window size (assumed square).
            num_heads (int): Number of attention heads.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            attn_drop (float): Dropout ratio on attention weights.
            proj_drop (float): Dropout ratio after projection.
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # This is fixed for the block
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Create a parameter table for relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        # Get pairwise relative position index for tokens inside a window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # (2, window_size, window_size)
        coords_flatten = torch.flatten(coords, 1)  # (2, window_size*window_size)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, window_size*window_size, window_size*window_size)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (window_size*window_size, window_size*window_size, 2)
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (window_size*window_size, window_size*window_size)
        self.register_buffer("relative_position_index", relative_position_index)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): Tensor of shape (num_windows*B, window_size*window_size, C)
            mask (torch.Tensor or None): Optional attention mask.
        
        Returns:
            x (torch.Tensor): Tensor of shape (num_windows*B, window_size*window_size, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, num_heads, N, N)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, N, N)
        attn = attn + relative_position_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock4D(nn.Module):
    """
    Swin Transformer block that accepts a 4D input tensor (B, C, H, W) and supports
    dynamic spatial sizes via on-the-fly padding. After processing, any added padding is removed.
    """
    def __init__(
        self, 
        dim, 
        num_heads, 
        window_size=7, 
        shift_size=0, 
        mlp_ratio=4.0, 
        qkv_bias=True, 
        dropout=0., 
        attn_drop=0., 
        drop_path=0.
    ):
        """
        Args:
            dim (int): Number of channels.
            num_heads (int): Number of attention heads.
            window_size (int): Fixed window size.
            shift_size (int): Shift size for cyclic shift (set to 0 for no shift).
            mlp_ratio (float): Ratio for the MLP hidden dimension.
            qkv_bias (bool): Whether to use bias in qkv layers.
            dropout (float): Dropout rate.
            attn_drop (float): Attention dropout rate.
            drop_path (float): Drop path rate (set to 0 or use an identity for simplicity).
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        
        # Token-based transformer layers (operating on flattened (B, H*W, C) tokens)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads,
                                      qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=dropout)
        self.drop_path = nn.Identity()  # Replace with a drop path module if desired
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, dropout=dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 4D input tensor of shape (B, C, H, W). H and W can be arbitrary.
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Dynamically pad input so that H and W become multiples of window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_r, 0, pad_b))  # Pad (left=0, right=pad_r, top=0, bottom=pad_b)
        H_pad, W_pad = H + pad_b, W + pad_r

        # Permute to (B, H_pad, W_pad, C) and flatten to tokens (B, H_pad*W_pad, C)
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        x_flat = x_perm.view(B, H_pad * W_pad, C)
        shortcut = x_flat

        # First normalization
        x_norm = self.norm1(x_flat)
        x_norm = x_norm.view(B, H_pad, W_pad, C)

        # Compute attention mask on the fly if using shifted windows
        if self.shift_size > 0:
            # Create an image mask of shape (1, H_pad, W_pad, 1)
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=x.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # (num_windows, window_size*window_size, 1)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        # Apply cyclic shift if shift_size > 0
        if self.shift_size > 0:
            shifted_x = torch.roll(x_norm, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_norm

        # Partition windows and apply window-based attention
        x_windows = window_partition(shifted_x, self.window_size)  # (num_windows*B, window_size*window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)          # (num_windows*B, window_size*window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H_pad, W_pad)  # (B, H_pad, W_pad, C)

        # Reverse cyclic shift if applied
        if self.shift_size > 0:
            x_out = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_out = shifted_x

        # Flatten tokens and apply residual connection and MLP
        x_out = x_out.view(B, H_pad * W_pad, C)
        x_out = shortcut + self.drop_path(x_out)
        x_out = x_out + self.drop_path(self.mlp(self.norm2(x_out)))
        # Reshape back to 4D (B, H_pad, W_pad, C) and then to (B, C, H_pad, W_pad)
        x_out = x_out.view(B, H_pad, W_pad, C).permute(0, 3, 1, 2).contiguous()

        # Remove any padding to restore original spatial dimensions
        if pad_r > 0 or pad_b > 0:
            x_out = x_out[:, :, :H, :W]
        return x_out

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
# Local Window Attention Module (original implementation)
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

# --------------------------------------------------
# SRBlock: Super Resolution Block with Two Parallel Paths
# --------------------------------------------------
class SRBlock(nn.Module):
    def __init__(self, channels, window_size=7):
        super(SRBlock, self).__init__()
        self.channels = channels
        self.window_size = window_size

        # Stage 1: Initial 3x3 Convolutions (with mirror padding) for two parallel branches.
        self.pad3 = nn.ReflectionPad2d(1)
        self.conv1_branch1 = nn.Conv2d(channels, channels, kernel_size=3, padding=0)
        self.conv1_branch2 = nn.Conv2d(channels, channels, kernel_size=3, padding=0)
        
        self.ln1_branch1 = nn.LayerNorm(channels)
        self.ln1_branch2 = nn.LayerNorm(channels)
        self.prelu = nn.PReLU()
        
        # Stage 1: LKA Module (from the paper)
        self.lka = LKA_Paper(channels)
        self.gamma1 = nn.Parameter(torch.ones(1))
        
        # Stage 2: Second 3x3 Convolutions for each branch.
        self.pad3_stage2 = nn.ReflectionPad2d(1)
        self.conv2_branch1 = nn.Conv2d(channels, channels, kernel_size=3, padding=0)
        self.conv2_branch2 = nn.Conv2d(channels, channels, kernel_size=3, padding=0)
        self.ln2_branch1 = nn.LayerNorm(channels)
        self.ln2_branch2 = nn.LayerNorm(channels)
        
        # --------------------------
        # Replace Local Window Attention with Swin Attention:
        # Original code:
        # self.lwa = LocalWindowAttention(channels, window_size)
        # self.gamma2 = nn.Parameter(torch.ones(1))
        # New code:
        self.swin_attn = SwinTransformerBlock4D(
            dim=channels,
            num_heads=4,             # adjust number of heads as needed
            window_size=window_size,
            shift_size=0,            # no cyclic shift for a direct replacement
            mlp_ratio=1.0,           # mlp_ratio=1.0 to avoid extra nonlinearity
            qkv_bias=True,
            dropout=0.0,
            attn_drop=0.0,
            drop_path=0.0
        )
        self.gamma2 = nn.Parameter(torch.ones(1))
        # --------------------------
        
        # Fusion Layer: 1x1 convolution to merge concatenated branches.
        self.conv_fusion = nn.Conv2d(channels * 2, channels, kernel_size=1)
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
        # Stage 2: Second Convolution and Swin Attention Interaction
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
        
        # ----------------------------
        # Swin Attention is applied separately to each branch.
        # Original Local Window Attention:
        # lwa_x2 = self.lwa(x2_2)
        # lwa_x1 = self.lwa(x1_2)
        # x1_2_hat = x1_2 + self.gamma2 * (lwa_x2 * x1_2)
        # x2_2_hat = x2_2 + self.gamma2 * (lwa_x1 * x2_2)
        # New Swin Attention usage:
        swin_x2 = self.swin_attn(x2_2)
        swin_x1 = self.swin_attn(x1_2)
        x1_2_hat = x1_2 + self.gamma2 * (swin_x2 * x1_2)
        x2_2_hat = x2_2 + self.gamma2 * (swin_x1 * x2_2)
        
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
