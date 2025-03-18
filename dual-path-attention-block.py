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
        self.scale = (channels // num_heads) ** -0.5
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.LayerNorm([channels])
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape to token format
        x_flat = x.flatten(2).transpose(1, 2)  # B, HW, C
        
        # Apply layer norm
        x_norm = self.norm(x_flat)
        x_norm = x_norm.transpose(1, 2).reshape(B, C, H, W)
        
        # Generate QKV
        qkv = self.qkv(x_norm).reshape(B, 3, self.num_heads, C // self.num_heads, H*W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention
        x_attn = (attn @ v).reshape(B, C, H, W)
        x_attn = self.proj(x_attn)
        
        # Convert to channel attention map
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
        self.scale = (channels // num_heads) ** -0.5
        
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Pad to multiple of window size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            _, _, H_pad, W_pad = x.shape
        else:
            H_pad, W_pad = H, W
        
        # Reshape to windows
        x_windows = x.reshape(B, C, H_pad // self.window_size, self.window_size, 
                              W_pad // self.window_size, self.window_size)
        x_windows = x_windows.permute(0, 2, 4, 3, 5, 1)  # B, H//ws, W//ws, ws, ws, C
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, C)  # (B*num_windows), ws*ws, C
        
        # Apply layer norm
        x_norm = self.norm(x_windows)
        
        # Multi-head self-attention
        qkv = self.qkv(x_norm).reshape(-1, self.window_size * self.window_size, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B*num_windows, num_heads, ws*ws, C//num_heads
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x_attn = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        x_attn = self.proj(x_attn)
        
        # Reshape back
        x_attn = x_attn.reshape(B, H_pad // self.window_size, W_pad // self.window_size, 
                               self.window_size, self.window_size, C)
        x_attn = x_attn.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H_pad, W_pad)
        
        # Unpad
        if pad_h > 0 or pad_w > 0:
            x_attn = x_attn[:, :, :H, :W]
        
        # Convert to spatial attention map
        attn_map = self.sigmoid(x_attn.mean(dim=1, keepdim=True))
        return attn_map

class DualPathAttentionBlock(nn.Module):
    def __init__(self, channels, reduction=16, window_size=8, num_heads=8):
        super().__init__()
        
        # Pathway 1 modules
        self.path1_stage1_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.path1_stage1_norm = nn.BatchNorm2d(channels)
        self.path1_stage1_act = nn.PReLU()
        
        self.path1_stage2_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.path1_stage2_norm = nn.BatchNorm2d(channels)
        self.path1_stage2_act = nn.PReLU()
        
        # Pathway 2 modules
        self.path2_stage1_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.path2_stage1_norm = nn.BatchNorm2d(channels)
        self.path2_stage1_act = nn.PReLU()
        
        self.path2_stage2_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.path2_stage2_norm = nn.BatchNorm2d(channels)
        self.path2_stage2_act = nn.PReLU()
        
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
        
        # Stage 1
        # Path 1 -> compute features and standard channel attention
        path1_stage1 = self.path1_stage1_conv(x)
        path1_stage1 = self.path1_stage1_norm(path1_stage1)
        path1_stage1 = self.path1_stage1_act(path1_stage1)
        
        channel_attn = self.std_channel_attn(path1_stage1)
        
        # Path 2 -> compute features and apply channel attention from Path 1
        path2_stage1 = self.path2_stage1_conv(x)
        path2_stage1 = self.path2_stage1_norm(path2_stage1)
        path2_stage1 = self.path2_stage1_act(path2_stage1)
        path2_stage1 = path2_stage1 * channel_attn
        
        # Stage 2
        # Path 2 -> compute features and transformer spatial attention
        path2_stage2 = self.path2_stage2_conv(path2_stage1)
        path2_stage2 = self.path2_stage2_norm(path2_stage2)
        path2_stage2 = self.path2_stage2_act(path2_stage2)
        
        spatial_attn = self.transformer_spatial_attn(path2_stage2)
        
        # Path 1 -> compute features and apply spatial attention from Path 2
        path1_stage2 = self.path1_stage2_conv(path1_stage1)
        path1_stage2 = self.path1_stage2_norm(path1_stage2)
        path1_stage2 = self.path1_stage2_act(path1_stage2)
        path1_stage2 = path1_stage2 * spatial_attn
        
        # Fusion
        concat_features = torch.cat([path1_stage2, path2_stage2], dim=1)
        fused_features = self.fusion_conv(concat_features)
        fused_features = self.fusion_norm(fused_features)
        fused_features = self.fusion_act(fused_features)
        
        # Residual connection
        output = self.final_norm(fused_features + identity)
        
        return output

# Example usage
if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    model = DualPathAttentionBlock(64)
    out = model(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
