import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return self.sigmoid(y)

class DualPathAttentionBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, reduction=16):
        super(DualPathAttentionBlock, self).__init__()
        
        # Path A convolutions
        self.path_a_conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.path_a_bn1 = nn.BatchNorm2d(channels)
        self.path_a_conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.path_a_bn2 = nn.BatchNorm2d(channels)
        
        # Path B convolutions
        self.path_b_conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.path_b_bn1 = nn.BatchNorm2d(channels)
        self.path_b_conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.path_b_bn2 = nn.BatchNorm2d(channels)
        
        # Attention mechanisms
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()
        
        # Fusion layer
        self.fusion = nn.Conv2d(2 * channels, channels, kernel_size=1)
        self.instance_norm = nn.InstanceNorm2d(channels)
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Initial paths
        path_a = x
        path_b = x
        
        # Stage 1
        path_a = self.relu(self.path_a_bn1(self.path_a_conv1(path_a)))
        path_b = self.relu(self.path_b_bn1(self.path_b_conv1(path_b)))
        
        # Channel attention from A modulates B
        channel_attn = self.channel_attention(path_a)
        path_b = path_b * channel_attn
        
        # Stage 2
        path_a = self.relu(self.path_a_bn2(self.path_a_conv2(path_a)))
        path_b = self.relu(self.path_b_bn2(self.path_b_conv2(path_b)))
        
        # Spatial attention from B modulates A
        spatial_attn = self.spatial_attention(path_b)
        path_a = path_a * spatial_attn
        
        # Fusion
        fused = torch.cat([path_a, path_b], dim=1)
        fused = self.fusion(fused)
        fused = self.instance_norm(fused)
        
        # Residual connection
        out = x + fused
        
        # Apply F.layer_norm directly instead of using nn.LayerNorm module
        # This handles dynamic input shapes
        out = F.layer_norm(out, [out.size(1), out.size(2), out.size(3)])
        
        return out


# Test code
def test_dual_path_attention_block():
    # Test parameters
    batch_size = 4
    channels = 64
    height, width = 64, 64
    
    # Create random input tensor
    x = torch.randn(batch_size, channels, height, width)
    
    # Create model
    model = DualPathAttentionBlock(channels)
    
    # Test forward pass
    output = model(x)
    
    # Test output shape
    assert output.shape == x.shape, f"Output shape {output.shape} doesn't match input shape {x.shape}"
    
    # Test with different input sizes
    new_height, new_width = 126, 126
    x_large = torch.randn(1, channels, new_height, new_width)
    output_large = model(x_large)
    assert output_large.shape == x_large.shape, f"Large: Output shape {output_large.shape} doesn't match input shape {x_large.shape}"
    
    print("All tests passed!")


if __name__ == "__main__":
    test_dual_path_attention_block()