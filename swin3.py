import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

# ----------------------- Test Code -----------------------
if __name__ == "__main__":
    # Simulate random conv feature map inputs with variable spatial sizes
    batch_size = 2
    channels = 96  # e.g. features from a conv layer
    # Create random spatial dimensions for testing
    spatial_sizes = [(56, 56), (63, 77), (32, 45)]
    for H, W in spatial_sizes:
        dummy_features = torch.randn(batch_size, channels, H, W)
        print(f"\nInput shape: {dummy_features.shape}")
        # Create the dynamic Swin block (with a cyclic shift example)
        swin_block = SwinTransformerBlock4D(
            dim=channels,
            num_heads=3,
            window_size=7,
            shift_size=3,  # Example: cyclic shift applied
            mlp_ratio=4.0,
            qkv_bias=True,
            dropout=0.1,
            attn_drop=0.1,
            drop_path=0.0
        )
        out_features = swin_block(dummy_features)
        print("Output shape:", out_features.shape)
