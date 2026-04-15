# ---------------------------------------
#        Generate Output (NO TRAINING)
# ---------------------------------------

import torch
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from torch import nn


# ---------------------------------------
#        Generator (same as training)
# ---------------------------------------
class Generator(nn.Module):
    def __init__(self, z_dim: int, img_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(True),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(True),
            nn.Linear(1024, img_dim), nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


# ---------------------------------------
#        Config
# ---------------------------------------
z_dim = 100
img_size = 28
img_dim = img_size * img_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = Path("./outputs/generator.pth")
output_path = Path("./outputs/final_output.png")


# ---------------------------------------
#        Load Model
# ---------------------------------------
print("Loading trained generator...")

gen = Generator(z_dim, img_dim).to(device)
gen.load_state_dict(torch.load(model_path, map_location=device))
gen.eval()

print("✅ Model loaded")


# ---------------------------------------
#        Generate Images
# ---------------------------------------
with torch.no_grad():
    noise = torch.randn(64, z_dim, device=device)
    fake_images = gen(noise)

print("Generating images...")


# ---------------------------------------
#        Save Image
# ---------------------------------------
save_image(
    fake_images.view(-1, 1, img_size, img_size),
    output_path,
    nrow=8,
    normalize=True
)

print(f"✅ Saved at {output_path}")


# ---------------------------------------
#        Show Image
# ---------------------------------------
grid = make_grid(
    fake_images.view(-1, 1, img_size, img_size),
    nrow=8,
    normalize=True
)

plt.imshow(grid.permute(1, 2, 0).cpu())
plt.title("Generated Images")
plt.axis('off')
plt.show()