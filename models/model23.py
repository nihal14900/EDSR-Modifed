import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#############################################
# Basic Building Blocks
#############################################

class ConvBNReLU(nn.Module):
    """A simple convolution followed by BatchNorm and ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ChannelAttentionMLP(nn.Module):
    """MLP for channel attention. Computes a per-channel modulation factor."""
    def __init__(self, channels, reduction=16):
        super(ChannelAttentionMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
    def forward(self, x):
        # x: (B, channels)
        return torch.sigmoid(self.mlp(x))


#############################################
# Stage 1: Normal Channel Attention
#############################################

class ChannelAttentionBlock(nn.Module):
    """
    Computes channel attention from a source feature map and modulates a target feature map.
    
    Given a source X (e.g., frequency branch), it performs global pooling, passes the result
    through an MLP, and then uses the obtained channel weights to modulate the target.
    """
    def __init__(self, channels, reduction=16):
        super(ChannelAttentionBlock, self).__init__()
        self.mlp  = ChannelAttentionMLP(channels, reduction)
        self.norm = nn.BatchNorm2d(channels)
    def forward(self, source, target):
        # source & target: (B, C, H, W)
        B, C, H, W = source.size()
        # Global average pooling on the source branch.
        gap = F.adaptive_avg_pool2d(source, 1).view(B, C)
        attn = self.mlp(gap).view(B, C, 1, 1)
        out  = target + target * attn
        out  = self.norm(out)
        return out


#############################################
# Stage 2: Transformer Channel Attention
#############################################

class TransformerChannelAttentionBlock(nn.Module):
    """
    For each branch, forms channel tokens via global pooling, embeds them, and processes
    them with a transformer encoder. The resulting tokens are projected back to scalars
    to modulate the original feature map.
    
    Returns both the modulated feature and the transformer tokens for later cross attention.
    """
    def __init__(self, channels, transformer_d=32, num_layers=1, num_heads=4):
        super(TransformerChannelAttentionBlock, self).__init__()
        self.channels      = channels
        self.transformer_d = transformer_d
        self.embed         = nn.Linear(1, transformer_d)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_d, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj        = nn.Linear(transformer_d, 1)
        self.norm        = nn.BatchNorm2d(channels)
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.size()
        gap = F.adaptive_avg_pool2d(x, 1).view(B, C)  # (B, C)
        # Embed each scalar into transformer_d dims.
        tokens = self.embed(gap.unsqueeze(-1))  # (B, C, transformer_d)
        # Transformer encoder expects (seq_len, batch, d)
        tokens = self.transformer(tokens.transpose(0, 1)).transpose(0, 1)  # (B, C, transformer_d)
        # Project back to scalar modulation factors.
        attn = torch.sigmoid(self.proj(tokens)).view(B, C, 1, 1)
        out  = x + x * attn
        out  = self.norm(out)
        return out, tokens  # Return tokens for subsequent cross attention


#############################################
# Cross Attention Module (for both Channel and Spatial)
#############################################

class CrossAttentionBlock(nn.Module):
    """
    A generic cross-attention module. Given query tokens from one branch and key/value tokens
    from another branch, it computes an attention-based modulation factor.
    
    The output has shape (B, seq_len, 1) and should be reshaped appropriately.
    """
    def __init__(self, transformer_d=32, num_heads=4):
        super(CrossAttentionBlock, self).__init__()
        self.query_proj = nn.Linear(transformer_d, transformer_d)
        self.key_proj   = nn.Linear(transformer_d, transformer_d)
        self.value_proj = nn.Linear(transformer_d, transformer_d)
        self.out_proj   = nn.Linear(transformer_d, 1)
        self.num_heads  = num_heads
    def forward(self, query_tokens, key_tokens, value_tokens):
        # query_tokens, key_tokens, value_tokens: (B, seq_len, d)
        Q = self.query_proj(query_tokens)  # (B, seq_len, d)
        K = self.key_proj(key_tokens)
        V = self.value_proj(value_tokens)
        d_k = Q.size(-1)
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)  # (B, seq_len, d)
        modulation = torch.sigmoid(self.out_proj(attn_out))  # (B, seq_len, 1)
        return modulation


#############################################
# Stage 3: Normal Spatial Attention
#############################################

class SpatialAttentionBlock(nn.Module):
    """
    Computes a spatial attention map from a source branch and uses it to modulate
    a target branch.
    """
    def __init__(self, channels):
        super(SpatialAttentionBlock, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.norm = nn.BatchNorm2d(channels)
    def forward(self, source, target):
        # source & target: (B, C, H, W)
        avg_pool = torch.mean(source, dim=1, keepdim=True)  # (B, 1, H, W)
        max_pool, _ = torch.max(source, dim=1, keepdim=True)  # (B, 1, H, W)
        concat = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, H, W)
        attn = torch.sigmoid(self.conv(concat))          # (B, 1, H, W)
        out  = target + target * attn
        out  = self.norm(out)
        return out


#############################################
# Stage 4: Transformer Spatial Attention
#############################################

class TransformerSpatialAttentionBlock(nn.Module):
    """
    Reshapes a feature map into spatial tokens, projects them to a transformer dimension,
    and applies window-based transformer self attention. The resulting tokens are projected
    to produce a scalar spatial modulation map.
    
    Returns both the modulated feature map and the tokens for cross attention.
    """
    def __init__(self, channels, transformer_d=32, num_layers=1, num_heads=4, window_size=(7,7)):
        super(TransformerSpatialAttentionBlock, self).__init__()
        self.channels      = channels
        self.transformer_d = transformer_d
        self.window_size   = window_size
        self.token_proj    = nn.Linear(channels, transformer_d)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_d, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj    = nn.Linear(transformer_d, 1)
        self.norm        = nn.BatchNorm2d(channels)
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.size()
        # Reshape to spatial tokens: (B, H*W, C)
        tokens = x.view(B, C, H*W).permute(0, 2, 1)  # (B, H*W, C)
        # Project tokens to transformer space: (B, H*W, transformer_d)
        tokens = self.token_proj(tokens)
        # Apply window-based self attention
        tokens = self.window_self_attention(tokens, H, W)
        # Project tokens to scalar modulation factors.
        attn = torch.sigmoid(self.out_proj(tokens))  # (B, H*W, 1)
        # Reshape to spatial modulation map: (B, 1, H, W)
        attn_map = attn.view(B, H, W, 1).permute(0, 3, 1, 2)
        out = x + x * attn_map
        out = self.norm(out)
        return out, tokens

    def window_self_attention(self, tokens, H, W):
        """
        Partitions spatial tokens into windows, applies transformer self-attention within each,
        and reassembles the tokens. Pads the feature map if necessary.
        """
        B, L, d = tokens.size()
        w_h, w_w = self.window_size

        # Compute required padding.
        pad_h = (w_h - H % w_h) % w_h
        pad_w = (w_w - W % w_w) % w_w
        H_pad = H + pad_h
        W_pad = W + pad_w

        # Reshape tokens to (B, H, W, d) and pad.
        tokens_2d = tokens.view(B, H, W, d)
        if pad_h > 0 or pad_w > 0:
            tokens_2d = F.pad(tokens_2d, (0, 0, 0, pad_w, 0, pad_h))
        
        # Partition into windows using unfold.
        tokens_windows = tokens_2d.unfold(1, w_h, w_h).unfold(2, w_w, w_w)
        B, num_windows_h, num_windows_w, _, _, d = tokens_windows.size()
        tokens_windows = tokens_windows.contiguous().view(B, num_windows_h * num_windows_w, w_h * w_w, d)
        # Merge windows into batch dimension.
        tokens_windows = tokens_windows.view(-1, w_h * w_w, d)  # (B*num_windows, window_area, d)
        # Transformer encoder expects (seq_len, batch, d)
        tokens_windows = self.transformer(tokens_windows.transpose(0,1)).transpose(0,1)
        # Reshape back to (B, H, W, d).
        tokens_windows = tokens_windows.view(B, num_windows_h * num_windows_w, w_h * w_w, d)
        tokens_windows = tokens_windows.view(B, num_windows_h, num_windows_w, w_h, w_w, d)
        tokens_2d = tokens_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_pad, W_pad, d)
        # Remove padding if added.
        if pad_h > 0 or pad_w > 0:
            tokens_2d = tokens_2d[:, :H, :W, :]
        tokens = tokens_2d.view(B, H*W, d)
        return tokens


#############################################
# Fusion Block: Merging Branches
#############################################

class FusionBlock(nn.Module):
    """
    Merges the two branches by concatenating along the channel dimension and
    applying a 1x1 convolution. A learnable scalar scales the fused output in
    a residual connection with the original input.
    """
    def __init__(self, channels):
        super(FusionBlock, self).__init__()
        self.conv_fuse = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.zeros(1))
    def forward(self, x1, x2, x_orig):
        merged = torch.cat([x1, x2], dim=1)
        fused  = self.conv_fuse(merged)
        return x_orig + self.alpha * fused


#############################################
# Full Dual-Domain Block (Modular)
#############################################

class DualDomainBlock(nn.Module):
    """
    Implements the full dual-domain block in a modular manner.
    
    It comprises:
      1. Branch Creation (frequency & spatial paths)
      2. Stage 1: Normal Channel Attention (bidirectional)
      3. Stage 2: Transformer Channel Attention with cross attention
      4. Stage 3: Normal Spatial Attention (bidirectional)
      5. Stage 4: Transformer Spatial Attention with cross attention
      6. Fusion with scaled residual connection
    """
    def __init__(self, channels, reduction=16, transformer_d=32, num_transformer_layers=1,
                 window_size_spatial=(7,7), num_heads=4):
        super(DualDomainBlock, self).__init__()
        self.channels = channels
        
        # Branch Creation
        self.branch_freq   = ConvBNReLU(channels, channels, kernel_size=3, padding=1)
        self.branch_spatial = ConvBNReLU(channels, channels, kernel_size=3, padding=1)
        
        # Stage 1: Normal Channel Attention (bidirectional)
        self.ca_freq_to_spatial   = ChannelAttentionBlock(channels, reduction)
        self.ca_spatial_to_freq   = ChannelAttentionBlock(channels, reduction)
        
        # Stage 2: Transformer Channel Attention for each branch
        self.tc_attn_freq   = TransformerChannelAttentionBlock(channels, transformer_d, num_transformer_layers, num_heads)
        self.tc_attn_spatial = TransformerChannelAttentionBlock(channels, transformer_d, num_transformer_layers, num_heads)
        # Cross Attention for channel tokens
        self.cross_attn_channel_freq   = CrossAttentionBlock(transformer_d, num_heads)
        self.cross_attn_channel_spatial = CrossAttentionBlock(transformer_d, num_heads)
        
        # Stage 3: Normal Spatial Attention (bidirectional)
        self.sa_freq_to_spatial   = SpatialAttentionBlock(channels)
        self.sa_spatial_to_freq   = SpatialAttentionBlock(channels)
        
        # Stage 4: Transformer Spatial Attention for each branch
        self.ts_attn_freq   = TransformerSpatialAttentionBlock(channels, transformer_d, num_transformer_layers, num_heads, window_size_spatial)
        self.ts_attn_spatial = TransformerSpatialAttentionBlock(channels, transformer_d, num_transformer_layers, num_heads, window_size_spatial)
        # Cross Attention for spatial tokens
        self.cross_attn_spatial_freq   = CrossAttentionBlock(transformer_d, num_heads)
        self.cross_attn_spatial_spatial = CrossAttentionBlock(transformer_d, num_heads)
        
        # Fusion
        self.fusion = FusionBlock(channels)
        
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.size()
        # Branch Creation
        X_f = self.branch_freq(x)      # Frequency branch
        X_s = self.branch_spatial(x)   # Spatial branch
        
        # Stage 1: Normal Channel Attention (bidirectional)
        Y_s = self.ca_freq_to_spatial(X_f, X_s)
        Y_f = self.ca_spatial_to_freq(X_s, X_f)
        
        # Stage 2: Transformer Channel Attention (self-attention per branch)
        X_f_tc, tokens_f = self.tc_attn_freq(Y_f)
        X_s_tc, tokens_s = self.tc_attn_spatial(Y_s)
        
        # Stage 2: Cross Attention for Channel Tokens
        # For frequency branch: use its tokens as query and spatial branch tokens as key/value.
        mod_factor_f = self.cross_attn_channel_freq(tokens_f, tokens_s, tokens_s)  # (B, C, 1)
        mod_factor_f = mod_factor_f.view(B, C, 1, 1)
        X_f_tc = self.tc_attn_freq.norm(X_f_tc + X_f_tc * mod_factor_f)
        
        # For spatial branch: use its tokens as query and frequency branch tokens as key/value.
        mod_factor_s = self.cross_attn_channel_spatial(tokens_s, tokens_f, tokens_f)  # (B, C, 1)
        mod_factor_s = mod_factor_s.view(B, C, 1, 1)
        X_s_tc = self.tc_attn_spatial.norm(X_s_tc + X_s_tc * mod_factor_s)
        
        # Stage 3: Normal Spatial Attention (bidirectional)
        Y_s_sa = self.sa_freq_to_spatial(X_f_tc, X_s_tc)
        Y_f_sa = self.sa_spatial_to_freq(X_s_tc, X_f_tc)
        
        # Stage 4: Transformer Spatial Attention (self-attention per branch)
        X_f_ts, tokens_f_sp = self.ts_attn_freq(Y_f_sa)
        X_s_ts, tokens_s_sp = self.ts_attn_spatial(Y_s_sa)
        
        # Stage 4: Cross Attention for Spatial Tokens
        mod_factor_f_sp = self.cross_attn_spatial_freq(tokens_f_sp, tokens_s_sp, tokens_s_sp)  # (B, H*W, 1)
        mod_factor_f_sp = mod_factor_f_sp.view(B, H, W, 1).permute(0, 3, 1, 2)
        X_f_ts = self.ts_attn_freq.norm(X_f_ts + X_f_ts * mod_factor_f_sp)
        
        mod_factor_s_sp = self.cross_attn_spatial_spatial(tokens_s_sp, tokens_f_sp, tokens_f_sp)  # (B, H*W, 1)
        mod_factor_s_sp = mod_factor_s_sp.view(B, H, W, 1).permute(0, 3, 1, 2)
        X_s_ts = self.ts_attn_spatial.norm(X_s_ts + X_s_ts * mod_factor_s_sp)
        
        # Fusion and Residual Connection
        out = self.fusion(X_f_ts, X_s_ts, x)
        return out


#############################################
# Test Code
#############################################

if __name__ == '__main__':
    # Create a random input tensor with odd spatial dimensions.
    B, C, H, W = 1, 64, 15, 17  # Height and width are odd.
    x = torch.randn(B, C, H, W)
    
    # Instantiate the dual-domain block.
    block = DualDomainBlock(
        channels=C, 
        reduction=16, 
        transformer_d=32,
        num_transformer_layers=1, 
        window_size_spatial=(3,3),  # Window size for spatial transformer attention.
        num_heads=4
    )
    
    # Forward pass.
    out = block(x)
    print("Input shape :", x.shape)
    print("Output shape:", out.shape)



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
            trunk.append(DualDomainBlock(
                channels=64, 
                reduction=16, 
                transformer_d=32,
                num_transformer_layers=1, 
                window_size_spatial=(3,3),  # Window size for spatial transformer attention.
                num_heads=4
            ))
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
