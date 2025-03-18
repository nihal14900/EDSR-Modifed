import torch
import torch.nn as nn
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis, parameter_count_table

# Import your model (EDSR or modified version)
from model14 import EDSR  
import config  # Ensure this contains upscale_factor and device

# Initialize the model
model = EDSR(config.upscale_factor).to(config.device)

# Print full model architecture
print("\n========== Model Architecture ==========")
print(model)

# Print total number of parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\n========== Model Parameters ==========")
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")

# Print layer-wise details
print("\n========== Layer-wise Summary ==========")
summary(model, input_size=(1, 3, 64, 64))  # Assuming input image size is 64x64

# Compute FLOPs
input_tensor = torch.randn(1, 3, 64, 64).to(config.device)  # Example input
flops = FlopCountAnalysis(model, input_tensor)
print("\n========== FLOPs (Floating Point Operations) ==========")
print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")  # Convert to GigaFLOPs

# Print parameter details using fvcore
print("\n========== Parameter Breakdown ==========")
print(parameter_count_table(model))
