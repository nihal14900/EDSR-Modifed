import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
from pathlib import Path

# Set the path to your image directory
image_dir = Path(r"E:\DF2K")

# Define transformation to scale images to [0,1]
transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32)  # Convert to float32 in range [0,1]
])

# Initialize accumulators for mean
mean_rgb = torch.zeros(3)
num_pixels = 0

# Iterate through images
for img_path in image_dir.glob("*.png"):  # Change extension if needed
    img = read_image(str(img_path))  # Read image as [C, H, W] tensor
    img = transform(img)  # Normalize to [0,1]

    mean_rgb += img.view(3, -1).mean(dim=1) * img.numel() // 3
    num_pixels += img.numel() // 3  # Count total pixels per channel

# Compute final mean per channel
mean_rgb /= num_pixels

print("Mean RGB values (scaled 1.0):", mean_rgb.tolist())

# DF2K: Mean RGB values (scaled 1.0): [0.46889835596084595, 0.4489615261554718, 0.40343502163887024]
