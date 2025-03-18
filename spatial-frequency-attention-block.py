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
        # Reshape to token format
        x_flat = x.flatten(2).transpose(1, 2)  # B, HW, C
        
        # Apply layer norm
        x_norm = self.norm(x_flat)
        x_norm = x_norm.transpose(1, 2).reshape(B, C, H, W)
        
        # Generate QKV
        qkv = self.qkv(x_norm).reshape(B, 3, self.num_heads, self.head_dim, H*W)
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
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Handle arbitrary input sizes
        num_h_windows = (H + self.window_size - 1) // self.window_size
        num_w_windows = (W + self.window_size - 1) // self.window_size
        
        # Pad to cover full windows
        H_pad = num_h_windows * self.window_size
        W_pad = num_w_windows * self.window_size
        padding_h = H_pad - H
        padding_w = W_pad - W
        
        if padding_h > 0 or padding_w > 0:
            x = F.pad(x, (0, padding_w, 0, padding_h))
        
        # Reshape to windows
        x_windows = x.reshape(B, C, num_h_windows, self.window_size, 
                             num_w_windows, self.window_size)
        x_windows = x_windows.permute(0, 2, 4, 3, 5, 1)  # B, num_h, num_w, ws, ws, C
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, C)  # (B*num_windows), ws*ws, C
        
        # Apply layer norm
        x_norm = self.norm(x_windows)
        
        # Multi-head self-attention
        qkv = self.qkv(x_norm).reshape(-1, self.window_size * self.window_size, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B*num_windows, num_heads, ws*ws, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x_attn = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        x_attn = self.proj(x_attn)
        
        # Reshape back
        x_attn = x_attn.reshape(B, num_h_windows, num_w_windows, 
                               self.window_size, self.window_size, C)
        x_attn = x_attn.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H_pad, W_pad)
        
        # Unpad
        if padding_h > 0 or padding_w > 0:
            x_attn = x_attn[:, :, :H, :W]
        
        # Convert to spatial attention map
        attn_map = self.sigmoid(x_attn.mean(dim=1, keepdim=True))
        return attn_map


class FrequencyDomainProcessing(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        # Learnable parameters for frequency domain
        self.magnitude_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.phase_conv = nn.Conv2d(channels*2, channels*2, kernel_size=1)
        self.magnitude_bn = nn.BatchNorm2d(channels)
        self.phase_bn = nn.BatchNorm2d(channels*2)
        self.act = nn.PReLU()
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Calculate output dimensions
        H_pad = ((H + 7) // 8) * 8
        W_pad = ((W + 7) // 8) * 8
        
        # Store original dtype for later conversion
        original_dtype = x.dtype
        
        # Pad input if necessary
        if H != H_pad or W != W_pad:
            x = F.pad(x, (0, W_pad - W, 0, H_pad - H))
        
        # Calculate expected number of patches
        patches_h = H_pad // 8
        patches_w = W_pad // 8
        num_patches = patches_h * patches_w
            
        # Unfold into patches (original dtype maintained)
        x_unf = F.unfold(x, kernel_size=8, stride=8, padding=0)  # B, C*64, num_patches
        
        # Process in spatial-frequency domain
        x_patches = x_unf.transpose(1, 2).reshape(B * num_patches, C, 8, 8)
        
        # Split into "low" and "high" frequency components
        x_low = F.avg_pool2d(x_patches, kernel_size=2, stride=2)  # B*patches, C, 4, 4
        x_low_upscaled = F.interpolate(x_low, size=(8, 8), mode='nearest')
        x_high = x_patches - x_low_upscaled
        
        # Process components (maintain original dtype)
        x_low = self.magnitude_conv(x_low)
        x_low = self.magnitude_bn(x_low)
        x_low = self.act(x_low)
        
        x_high_stacked = torch.cat([x_high, torch.roll(x_high, shifts=1, dims=2)], dim=1)
        x_high_stacked = self.phase_conv(x_high_stacked)
        x_high_stacked = self.phase_bn(x_high_stacked)
        x_high_stacked = self.act(x_high_stacked)
        x_high = x_high_stacked[:, :C]
        
        # Combine components
        x_low_final = F.interpolate(x_low, size=(8, 8), mode='nearest')
        x_proc = x_low_final + x_high  # B*patches, C, 8, 8
        
        # Reshape to format expected by fold operation
        x_proc = x_proc.reshape(B, num_patches, -1)  # B, patches, C*64
        x_proc = x_proc.transpose(1, 2)  # B, C*64, patches
        
        # Fold back to image - should match expected patch count now
        output_size = (H_pad, W_pad)
        x_output = F.fold(x_proc, output_size=output_size, kernel_size=8, stride=8)
        
        # Crop back to original size if needed
        if H != H_pad or W != W_pad:
            x_output = x_output[:, :, :H, :W]
            
        return x_output


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
        
        # Frequency domain pathway
        self.freq_domain_stage1 = FrequencyDomainProcessing(channels)
        self.freq_stage1_norm = nn.BatchNorm2d(channels)
        self.freq_stage1_act = nn.PReLU()
        
        self.freq_domain_stage2 = FrequencyDomainProcessing(channels)
        self.freq_stage2_norm = nn.BatchNorm2d(channels)
        self.freq_stage2_act = nn.PReLU()
        
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
        # Spatial pathway -> compute features and standard channel attention
        spatial_stage1 = self.spatial_stage1_conv(x)
        spatial_stage1 = self.spatial_stage1_norm(spatial_stage1)
        spatial_stage1 = self.spatial_stage1_act(spatial_stage1)
        
        channel_attn = self.std_channel_attn(spatial_stage1)
        
        # Frequency pathway -> compute features and apply channel attention from spatial path
        freq_stage1 = self.freq_domain_stage1(x)
        freq_stage1 = self.freq_stage1_norm(freq_stage1)
        freq_stage1 = self.freq_stage1_act(freq_stage1)
        freq_stage1 = freq_stage1 * channel_attn
        
        # Stage 2
        # Frequency pathway -> compute features and transformer spatial attention
        freq_stage2 = self.freq_domain_stage2(freq_stage1)
        freq_stage2 = self.freq_stage2_norm(freq_stage2)
        freq_stage2 = self.freq_stage2_act(freq_stage2)
        
        spatial_attn = self.transformer_spatial_attn(freq_stage2)
        
        # Spatial pathway -> compute features and apply spatial attention from frequency path
        spatial_stage2 = self.spatial_stage2_conv(spatial_stage1)
        spatial_stage2 = self.spatial_stage2_norm(spatial_stage2)
        spatial_stage2 = self.spatial_stage2_act(spatial_stage2)
        spatial_stage2 = spatial_stage2 * spatial_attn
        
        # Fusion
        concat_features = torch.cat([spatial_stage2, freq_stage2], dim=1)
        fused_features = self.fusion_conv(concat_features)
        fused_features = self.fusion_norm(fused_features)
        fused_features = self.fusion_act(fused_features)
        
        # Residual connection
        output = self.final_norm(fused_features + identity)
        
        return output


# Test function to verify the model works with various input sizes
def test_model():
    model = SpatialFrequencyAttentionBlock(64)
    
    # Test with power of two size
    x = torch.randn(2, 64, 32, 32)
    out = model(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    
    # Test with odd size
    x = torch.randn(2, 64, 33, 33)
    out = model(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    
    # Test with arbitrary size
    x = torch.randn(2, 64, 37, 41)
    out = model(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    
    # Test with half precision
    if torch.cuda.is_available():
        model = model.cuda().half()
        x = torch.randn(2, 64, 48, 48).cuda().half()
        out = model(x)
        print(f"Half precision input shape: {x.shape}, Output shape: {out.shape}")


if __name__ == "__main__":
    test_model()