import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Stage 1: Normal Channel Attention
# ------------------------------
class NormalChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(NormalChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # output (B, C, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=True),
            nn.Sigmoid()
        )
        # BatchNorm2d here stabilizes the recalibrated features
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.size()
        y = self.avg_pool(x).view(B, C)      # (B, C)
        y = self.fc(y).view(B, C, 1, 1)        # (B, C, 1, 1)
        # Multiply input by the learned weights and add a residual connection
        out = x + x * y
        out = self.norm(out)
        return out

# ------------------------------
# Stage 3: Normal Spatial Attention
# ------------------------------
class NormalSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(NormalSpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel pooling: compute both average and max along channel axis
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        attn = self.sigmoid(self.conv(x_cat))          # (B, 1, H, W)
        out = x + x * attn
        return out

# ------------------------------
# Cross Attention Modules
# ------------------------------
# (a) Cross Channel Attention – a simple MLP on pooled (channel) descriptors
class CrossChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CrossChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x_query, x_key):
        # x_query, x_key: (B, C, H, W)
        B, C, H, W = x_query.size()
        query_avg = self.avg_pool(x_query).view(B, C)  # (B, C)
        key_avg = self.avg_pool(x_key).view(B, C)        # (B, C)
        combined = torch.cat([query_avg, key_avg], dim=1)  # (B, 2C)
        weights = self.fc(combined).view(B, C, 1, 1)       # (B, C, 1, 1)
        out = x_query + x_query * weights
        return out

# (b) Cross Spatial Attention – using nn.MultiheadAttention
class CrossSpatialAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(CrossSpatialAttention, self).__init__()
        # Use batch_first=True so that inputs are (B, L, E)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x_query, x_key):
        # x_query, x_key: (B, C, H, W)
        B, C, H, W = x_query.shape
        L = H * W
        # Reshape: treat each spatial position as a token.
        q = x_query.view(B, C, L).permute(0, 2, 1)  # (B, L, C)
        k = x_key.view(B, C, L).permute(0, 2, 1)      # (B, L, C)
        # Use x_key for both key and value
        attn_out, _ = self.mha(q, k, k)               # (B, L, C)
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        return attn_out

# ------------------------------
# Stage 2: Cross Transformer Channel Attention (CTCA)
# ------------------------------
class CTCA(nn.Module):
    def __init__(self, channels, d_model=64, nhead=4):
        super(CTCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # To form channel tokens
        # Map each channel’s scalar to a d_model-dim token.
        self.fc_in = nn.Linear(1, d_model)
        # Transformer encoder layer (note: sequence length = channels)
        self.transformer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        # Project back to a scalar per channel.
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.size()
        # Global average pooling: get one value per channel
        tokens = self.avg_pool(x).view(B, C, 1)  # (B, C, 1)
        # Rearrange tokens to shape (C, B, 1) for the transformer (sequence length = C)
        tokens = tokens.transpose(0, 1)          # (C, B, 1)
        tokens = self.fc_in(tokens)              # (C, B, d_model)
        tokens = self.transformer(tokens)         # (C, B, d_model)
        tokens = self.fc_out(tokens)              # (C, B, 1)
        # Rearrange back to (B, C, 1, 1)
        weights = tokens.transpose(0, 1).unsqueeze(-1)  # (B, C, 1, 1)
        out = x + x * weights
        return out

# ------------------------------
# Stage 4: Cross Transformer Spatial Attention (CTSA)
# ------------------------------
class CTSA(nn.Module):
    def __init__(self, channels, nhead=4, dropout=0.0):
        super(CTSA, self).__init__()
        # Here the tokens are spatial positions; d_model equals number of channels.
        self.transformer = nn.TransformerEncoderLayer(d_model=channels, nhead=nhead, dropout=dropout, batch_first=True)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.size()
        L = H * W
        # Reshape so that each spatial location is a token
        tokens = x.view(B, C, L).permute(0, 2, 1)  # (B, L, C)
        out = self.transformer(tokens)              # (B, L, C)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return x + out

# ------------------------------
# Dual-Domain Block
# ------------------------------
class DualDomainBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, d_model=64, nhead=4, dropout=0.1):
        super(DualDomainBlock, self).__init__()
        # Branch splitting: separate conv layers for frequency and spatial paths.
        self.conv_freq = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_spatial = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        # Stage 1: Normal Channel Attention
        self.ca = NormalChannelAttention(in_channels, reduction)
        # Cross Attention after Stage 1 (channel exchange)
        self.cross_ca = CrossChannelAttention(in_channels, reduction)
        
        # Stage 2: Cross Transformer Channel Attention (CTCA)
        self.ctca = CTCA(in_channels, d_model, nhead)
        # Cross Attention after Stage 2 (channel)
        self.cross_ca2 = CrossChannelAttention(in_channels, reduction)
        
        # Stage 3: Normal Spatial Attention
        self.sa = NormalSpatialAttention(kernel_size=7)
        # Cross Attention after Stage 3 (spatial exchange)
        self.cross_sa = CrossSpatialAttention(in_channels, num_heads=nhead, dropout=dropout)
        
        # Stage 4: Cross Transformer Spatial Attention (CTSA)
        self.ctsa = CTSA(in_channels, nhead, dropout)
        # Cross Attention after Stage 4 (spatial)
        self.cross_sa2 = CrossSpatialAttention(in_channels, num_heads=nhead, dropout=dropout)
        
        # Merge branches: concatenate along channel dimension and fuse with a 1x1 conv.
        self.merge_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        
        # Scaled residual connection parameter.
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, H, W) where H and W can be non-even.
        x_act = F.relu(x)
        # Split into two branches with distinct convolutional filters.
        branch_freq = F.relu(self.conv_freq(x_act))
        branch_spat = F.relu(self.conv_spatial(x_act))
        
        # ----- Stage 1: Normal Channel Attention -----
        branch_freq = self.ca(branch_freq)
        branch_spat = self.ca(branch_spat)
        # Cross attention exchange after Stage 1 (channel domain)
        branch_freq = self.cross_ca(branch_freq, branch_spat)
        branch_spat = self.cross_ca(branch_spat, branch_freq)
        
        # ----- Stage 2: Cross Transformer Channel Attention (CTCA) -----
        branch_freq = self.ctca(branch_freq)
        branch_spat = self.ctca(branch_spat)
        # Cross attention exchange after Stage 2 (channel domain)
        branch_freq = self.cross_ca2(branch_freq, branch_spat)
        branch_spat = self.cross_ca2(branch_spat, branch_freq)
        
        # ----- Stage 3: Normal Spatial Attention -----
        branch_freq = self.sa(branch_freq)
        branch_spat = self.sa(branch_spat)
        # Cross attention exchange after Stage 3 (spatial domain)
        cross_spat_freq = self.cross_sa(branch_freq, branch_spat)
        cross_spat_spat = self.cross_sa(branch_spat, branch_freq)
        branch_freq = branch_freq + self.dropout(cross_spat_freq)
        branch_spat = branch_spat + self.dropout(cross_spat_spat)
        
        # ----- Stage 4: Cross Transformer Spatial Attention (CTSA) -----
        branch_freq = self.ctsa(branch_freq)
        branch_spat = self.ctsa(branch_spat)
        # Cross attention exchange after Stage 4 (spatial domain)
        cross_spat_freq2 = self.cross_sa2(branch_freq, branch_spat)
        cross_spat_spat2 = self.cross_sa2(branch_spat, branch_freq)
        branch_freq = branch_freq + self.dropout(cross_spat_freq2)
        branch_spat = branch_spat + self.dropout(cross_spat_spat2)
        
        # ----- Merge and Scaled Residual Connection -----
        merged = torch.cat([branch_freq, branch_spat], dim=1)  # (B, 2*C, H, W)
        merged = self.merge_conv(merged)                       # (B, C, H, W)
        out = x + self.alpha * merged
        return out

# ------------------------------
# Test Code
# ------------------------------
if __name__ == "__main__":
    # Create a random input tensor with non-even height and width (e.g., H=31, W=45)
    B = 1
    C = 64
    H = 31
    W = 45
    x = torch.randn(B, C, H, W)
    
    # Instantiate the block and run a forward pass.
    block = DualDomainBlock(in_channels=C)
    out = block(x)
    
    print("Input shape: ", x.shape)
    print("Output shape:", out.shape)
