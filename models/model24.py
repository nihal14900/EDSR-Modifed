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
    
import math

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
            trunk.append(DualPathAttentionBlock(64))
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
