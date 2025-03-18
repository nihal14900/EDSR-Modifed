import torch
import torch.nn as nn
import torch.nn.functional as F

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
        x_reshaped = x.view(1, B * C, H, W)
        # 2. Reshape aggregated_weight from (B, out_channels, C // groups, kH, kW)
        #    to (B * out_channels, C // groups, kH, kW)
        B, Cout, Cin_per_group, kH, kW = aggregated_weight.shape
        weight_reshaped = aggregated_weight.view(B * Cout, Cin_per_group, kH, kW)
        
        # Reshape bias if available
        if aggregated_bias is not None:
            bias_reshaped = aggregated_bias.view(B * Cout)
        else:
            bias_reshaped = None
        
        # Perform convolution with groups = B * groups
        out = F.conv2d(x_reshaped, weight_reshaped, bias_reshaped,
                       stride=self.stride, padding=self.padding, 
                       dilation=self.dilation, groups=B * self.groups)
        # Reshape output back to (B, out_channels, H_out, W_out)
        out = out.view(B, Cout, out.shape[-2], out.shape[-1])
        return out

# Example usage:
if __name__ == '__main__':
    # Create a random input tensor with shape (batch, channels, height, width)
    x = torch.randn(8, 64, 32, 32)
    # Create a dynamic convolution layer similar to nn.Conv2d
    dyn_conv = DynamicConv2d(in_channels=64, out_channels=128, kernel_size=3, 
                             stride=1, padding=1, K=4, reduction=4, temperature=30)
    # Forward pass
    y = dyn_conv(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
