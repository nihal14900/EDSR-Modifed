import torch
import torch.nn as nn
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from model13 import EDSR

def compute_model_statistics(model, input_size=(1, 3, 224, 224)):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Model summary using torchinfo
    model_summary = summary(model, input_size=input_size, verbose=0, device=device)

    # Total Parameters
    total_params = model_summary.total_params
    trainable_params = model_summary.trainable_params
    non_trainable_params = total_params - trainable_params

    # FLOPs estimation using fvcore (ensuring input is on the same device)
    inputs = torch.randn(input_size).to(device)
    flops = FlopCountAnalysis(model, inputs)

    # Memory estimation (assuming 4 bytes per parameter)
    param_memory = total_params * 4 / (1024 ** 2)  # MB

    print("="*50)
    print(f"Device: {device}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-Trainable Parameters: {non_trainable_params:,}")
    print(f"Estimated Parameter Memory Usage: {param_memory:.2f} MB")
    print(f"FLOPs per Forward Pass: {flops.total():,}")
    print("="*50)

    print("\nLayer-wise Parameter Statistics:")
    print(parameter_count_table(model))

# Example usage with a large model
large_model = EDSR(4)
compute_model_statistics(large_model, input_size=(1, 3, 64, 64))
