import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def window_partition(x, window_size):
    """
    Partition the input feature map x into non-overlapping windows.
    
    Args:
        x (torch.Tensor): Tensor of shape (B, H, W, C)
        window_size (int): The window size.
    
    Returns:
        windows (torch.Tensor): Tensor of shape (num_windows*B, window_size*window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # Permute and reshape to (num_windows*B, window_size*window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Reverse the window partitioning to reconstruct the original feature map.
    
    Args:
        windows (torch.Tensor): Tensor of shape (num_windows*B, window_size*window_size, C)
        window_size (int): Window size.
        H (int): Height of the feature map.
        W (int): Width of the feature map.
    
    Returns:
        x (torch.Tensor): Reconstructed tensor of shape (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Mlp(nn.Module):
    """
    Multilayer Perceptron as used in Transformer blocks.
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
    Window based multi-head self attention (W-MSA) with relative positional bias.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        """
        Args:
            dim (int): Number of input channels.
            window_size (int): Size of the window (assumed square).
            num_heads (int): Number of attention heads.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            attn_drop (float): Dropout ratio for attention weights.
            proj_drop (float): Dropout ratio after projection.
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        # get pairwise relative position index for each token inside the window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        # Use torch.meshgrid with indexing='ij' for row, col order
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # shape (2, window_size, window_size)
        coords_flatten = torch.flatten(coords, 1)  # shape (2, window_size*window_size)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # shape (2, window_size*window_size, window_size*window_size)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # shape (window_size*window_size, window_size*window_size, 2)
        relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # shape (window_size*window_size, window_size*window_size)
        self.register_buffer("relative_position_index", relative_position_index)
        
        # initialize the relative position bias table
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): Input features of shape (num_windows*B, N, C), where N = window_size*window_size.
            mask (torch.Tensor or None): (0/-100) mask with shape (num_windows, N, N) or None.
        Returns:
            x (torch.Tensor): Output features after window attention.
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # shape: (3, B_, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # shape: (B_, num_heads, N, N)
        
        # add relative positional bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1)  # shape: (N, N, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # shape: (1, num_heads, N, N)
        attn = attn + relative_position_bias
        
        # apply attention mask if provided
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

class SwinTransformerBlock(nn.Module):
    """
    A Swin Transformer block that applies window-based multi-head self-attention (with optional cyclic shift)
    followed by an MLP. This block includes layer normalization and residual connections.
    """
    def __init__(
        self, 
        dim, 
        input_resolution, 
        num_heads, 
        window_size=7, 
        shift_size=0, 
        mlp_ratio=4., 
        qkv_bias=True, 
        dropout=0., 
        attn_drop=0., 
        drop_path=0.
    ):
        """
        Args:
            dim (int): Number of input channels.
            input_resolution (tuple[int, int]): Input resolution (H, W).
            num_heads (int): Number of attention heads.
            window_size (int): Window size. If input resolution is smaller than window_size, it is adjusted.
            shift_size (int): Shift size for cyclic shifting (0 for no shift).
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to qkv.
            dropout (float): Dropout rate after projections.
            attn_drop (float): Dropout rate for attention weights.
            drop_path (float): Stochastic depth rate (here replaced by Identity for simplicity).
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (H, W)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        if min(input_resolution) <= window_size:
            # adjust window size if input is smaller than the set window size
            self.window_size = min(input_resolution)
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, 
            window_size=self.window_size, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            attn_drop=attn_drop, 
            proj_drop=dropout
        )
        # For simplicity, drop_path is set as Identity. Replace with a DropPath module if needed.
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, dropout=dropout)
        
        # Create attention mask for shifted windows if needed
        if self.shift_size > 0:
            H, W = self.input_resolution
            # create an image mask
            img_mask = torch.zeros((1, H, W, 1))
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
            # partition the mask into windows
            mask_windows = window_partition(img_mask, self.window_size)  # (num_windows, window_size*window_size, 1)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            # compute attention mask: 0 for same window, -100.0 for different windows
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, H*W, C)
        Returns:
            x (torch.Tensor): Output tensor of shape (B, H*W, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # cyclic shift if shift_size > 0
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # partition windows and apply window attention
        x_windows = window_partition(shifted_x, self.window_size)  # (num_windows*B, window_size*window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # (num_windows*B, window_size*window_size, C)
        # merge windows back to feature map
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # (B, H, W, C)
        
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        
        # apply residual connections and MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# ----------------------- Test Code -----------------------
if __name__ == "__main__":
    # Test parameters
    batch_size = 2
    H, W = 56, 56       # input resolution (height, width)
    embed_dim = 96      # embedding dimension
    num_heads = 3       # number of attention heads
    window_size = 7     # window size
    # Create dummy input with shape (B, H*W, C)
    x = torch.randn(batch_size, H * W, embed_dim)
    
    # Test Swin Transformer block without shift (shift_size=0)
    print("Testing non-shifted window attention:")
    block = SwinTransformerBlock(
        dim=embed_dim, 
        input_resolution=(H, W), 
        num_heads=num_heads, 
        window_size=window_size, 
        shift_size=0, 
        mlp_ratio=4.0, 
        qkv_bias=True, 
        dropout=0.1, 
        attn_drop=0.1, 
        drop_path=0.0
    )
    out = block(x)
    print("Output shape (no shift):", out.shape)
    
    # Test Swin Transformer block with shifted windows (shift_size > 0)
    print("\nTesting shifted window attention:")
    shift_size = window_size // 2  # a typical setting for shifted windows
    block_shift = SwinTransformerBlock(
        dim=embed_dim, 
        input_resolution=(H, W), 
        num_heads=num_heads, 
        window_size=window_size, 
        shift_size=shift_size, 
        mlp_ratio=4.0, 
        qkv_bias=True, 
        dropout=0.1, 
        attn_drop=0.1, 
        drop_path=0.0
    )
    out_shift = block_shift(x)
    print("Output shape (with shift):", out_shift.shape)


# def compute_memory_consumption(model, input_tensor):
#     """
#     Estimate memory consumption (in MB) for a model's parameters and activations.
    
#     Args:
#         model (torch.nn.Module): The PyTorch model/module.
#         input_tensor (torch.Tensor): A sample input tensor to run a forward pass.
    
#     Returns:
#         param_mem_mb (float): Memory used by model parameters in MB.
#         activation_mem_mb (float): Estimated memory used by activations during forward pass in MB.
#     """
#     # Compute memory for model parameters (in bytes)
#     param_mem_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    
#     # This variable will accumulate the memory usage of outputs (activations) from all modules.
#     activation_mem_bytes = 0

#     # Define a hook that adds the memory of the module's output.
#     def hook_fn(module, input, output):
#         nonlocal activation_mem_bytes
#         if isinstance(output, torch.Tensor):
#             activation_mem_bytes += output.numel() * output.element_size()
#         elif isinstance(output, (list, tuple)):
#             for o in output:
#                 if isinstance(o, torch.Tensor):
#                     activation_mem_bytes += o.numel() * o.element_size()

#     # Register the hook to every module in the model.
#     hooks = []
#     for module in model.modules():
#         hooks.append(module.register_forward_hook(hook_fn))
    
#     # Run a forward pass to collect activation sizes.
#     with torch.no_grad():
#         _ = model(input_tensor)
    
#     # Remove all hooks.
#     for h in hooks:
#         h.remove()
    
#     # Convert bytes to megabytes (MB)
#     param_mem_mb = param_mem_bytes / (1024 ** 2)
#     activation_mem_mb = activation_mem_bytes / (1024 ** 2)
    
#     return param_mem_mb, activation_mem_mb

# # ----------------------- Test Code for Memory Consumption -----------------------
# if __name__ == "__main__":
#     import torch
#     # Assume SwinTransformerBlock (from previous code) is already defined/imported.

#     # Test parameters
#     batch_size = 2
#     H, W = 56, 56       # input resolution (height, width)
#     embed_dim = 96      # embedding dimension
#     num_heads = 3       # number of attention heads
#     window_size = 7     # window size
    
#     # Create dummy input with shape (B, H*W, C)
#     x = torch.randn(batch_size, H * W, embed_dim)
    
#     # Create an instance of the Swin Transformer block (using non-shifted windows as an example)
#     block = SwinTransformerBlock(
#         dim=embed_dim, 
#         input_resolution=(H, W), 
#         num_heads=num_heads, 
#         window_size=window_size, 
#         shift_size=0,  # no shift
#         mlp_ratio=4.0, 
#         qkv_bias=True, 
#         dropout=0.1, 
#         attn_drop=0.1, 
#         drop_path=0.0
#     )
    
#     # Compute memory consumption
#     param_mem, activation_mem = compute_memory_consumption(block, x)
#     print(f"Parameter memory consumption: {param_mem:.2f} MB")
#     print(f"Activation memory consumption (approx.): {activation_mem:.2f} MB")