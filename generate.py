# ---------------------------------------
#        Generate Output (NO TRAINING)
# ---------------------------------------
"""
Inference script for a trained GAN Generator.

This script loads a pre-trained Generator model and uses it to
generate new images from random noise (latent vectors).

Workflow:
    1. Load trained Generator weights
    2. Sample random noise vectors
    3. Generate fake images
    4. Save and visualize the output

Important:
    - No training occurs in this script
    - Only forward pass (inference) is performed
"""

import torch
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from torch import nn


# ---------------------------------------
#        Generator (same as training)
# ---------------------------------------
class Generator(nn.Module):
    """
    Generator network used during inference.

    This must exactly match the architecture used during training,
    otherwise the saved weights cannot be loaded correctly.

    Purpose:
        Convert random noise (latent vectors) into synthetic images.

    Architecture:
        z_dim → 256 → 512 → 1024 → img_dim

    Output:
        Flattened image tensor with values in [-1, 1]
    """
    def __init__(self, z_dim: int, img_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(True),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(True),
            nn.Linear(1024, img_dim), nn.Tanh(),
        )

    def forward(self, z):
        """
        Forward pass of Generator.

        Args:
            z (torch.Tensor): Random noise vector

        Returns:
            torch.Tensor: Generated fake images
        """
        return self.model(z)


# ---------------------------------------
#        Config
# ---------------------------------------
"""
Configuration for inference.

Defines:
    - Latent vector size (z_dim)
    - Image size
    - Device (CPU/GPU)
    - Model path (saved Generator)
    - Output image path
"""

z_dim = 100
img_size = 28
img_dim = img_size * img_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = Path("./outputs/generator.pth")
output_path = Path("./outputs/final_output.png")


# ---------------------------------------
#        Load Model
# ---------------------------------------
"""
Load the trained Generator model from disk.

Steps:
    1. Initialize Generator architecture
    2. Load saved weights
    3. Set model to evaluation mode

Note:
    eval() disables BatchNorm updates and ensures consistent output.
"""

print("Loading trained generator...")

gen = Generator(z_dim, img_dim).to(device)
gen.load_state_dict(torch.load(model_path, map_location=device))
gen.eval()

print("✅ Model loaded successfully")


# ---------------------------------------
#        Generate Images
# ---------------------------------------
"""
Generate fake images using the trained Generator.

Steps:
    1. Sample random noise from normal distribution
    2. Pass noise through Generator
    3. Produce synthetic images

Note:
    torch.no_grad() disables gradient computation for faster inference.
"""

with torch.no_grad():
    noise = torch.randn(64, z_dim, device=device)
    fake_images = gen(noise)

print("Generating images...")


# ---------------------------------------
#        Save Image
# ---------------------------------------
"""
Save generated images to disk.

Steps:
    - Reshape flattened images into (C, H, W)
    - Normalize values from [-1,1] to [0,1]
    - Save as grid image

Output:
    outputs/final_output.png
"""

save_image(
    fake_images.view(-1, 1, img_size, img_size),
    output_path,
    nrow=8,
    normalize=True
)

print(f"✅ Image saved at: {output_path}")


# ---------------------------------------
#        Show Image
# ---------------------------------------
"""
Display generated images using matplotlib.

Steps:
    - Create grid using torchvision
    - Convert tensor to displayable format
    - Show image

Result:
    A grid of AI-generated handwritten digits
"""

grid = make_grid(
    fake_images.view(-1, 1, img_size, img_size),
    nrow=8,
    normalize=True
)

plt.imshow(grid.permute(1, 2, 0).cpu())
plt.title("Generated Images")
plt.axis('off')
plt.show()