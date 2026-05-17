# ---------------------------------------
#               Imports
# ---------------------------------------
"""
Basic GAN implementation using PyTorch.

This script trains a Generative Adversarial Network (GAN) on the MNIST dataset.
The GAN consists of:
    - Generator: learns to create realistic handwritten digits from noise
    - Discriminator: learns to distinguish real vs fake images

Workflow:
    1. Load and preprocess MNIST dataset
    2. Train Generator and Discriminator in an adversarial manner
    3. Visualize generated images during training
    4. Save trained models

This implementation is suitable for learning GAN fundamentals.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets, utils
from tqdm import tqdm
import matplotlib.pyplot as plt


# ---------------------------------------
#            Utility Functions
# ---------------------------------------

def weights_init(m: nn.Module):
    """
    Initializes the weights of neural network layers following DCGAN guidelines.

    This function is applied to each layer of the Generator and Discriminator
    to ensure stable GAN training.

    Initialization Strategy:
    - Weights are sampled from a normal distribution with mean=0 and std=0.02
    - Bias terms (if present) are initialized to 0

    Args:
        m (nn.Module): A layer/module in the network.

    Note:
        Proper weight initialization is critical in GANs to avoid instability
        such as mode collapse or vanishing gradients.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def show(tensor: torch.Tensor, ch: int = 1, size: tuple = (28, 28), num: int = 16):
    """
    Visualizes a batch of images (real or generated) in a grid format.

    Steps performed:
    1. Detaches tensor from computation graph and moves it to CPU
    2. Reshapes flattened images into (C, H, W) format
    3. Converts pixel values from [-1, 1] → [0, 1] for visualization
    4. Creates a grid of images using torchvision utilities
    5. Displays the images using matplotlib

    Args:
        tensor (torch.Tensor): Batch of images (flattened or generated)
        ch (int): Number of channels (1 for grayscale, 3 for RGB)
        size (tuple): Image dimensions (height, width)
        num (int): Number of images to display

    Note:
        GAN outputs use Tanh activation → range [-1, 1],
        so normalization is required before visualization.
    """
    data = tensor.detach().cpu().view(-1, ch, *size)
    data = (data + 1) / 2
    data = data.clamp(0, 1)

    grid = utils.make_grid(data[:num], nrow=4)
    grid = grid.permute(1, 2, 0)

    plt.figure()
    if ch == 1:
        plt.imshow(grid.squeeze(), cmap='gray')
    else:
        plt.imshow(grid)
    plt.axis('off')
    plt.show()


# ---------------------------------------
#               Configuration
# ---------------------------------------

@dataclass
class GANConfig:
    """
    Configuration class for GAN training.

    This dataclass centralizes all hyperparameters and paths used
    during training, making the experiment reproducible and easy to modify.

    Attributes:
        data_dir (Path): Directory where dataset is stored/downloaded
        output_dir (Path): Directory to save models and logs
        batch_size (int): Number of samples per training batch
        z_dim (int): Dimension of latent noise vector (input to Generator)
        lr (float): Learning rate for both Generator and Discriminator
        betas (tuple): Adam optimizer parameters (momentum terms)
        epochs (int): Number of full training iterations over dataset
        img_size (int): Height/width of input images (MNIST = 28)
        device (torch.device): CPU or GPU device
        log_interval (int): Steps between logging and visualization

    Note:
        Using a config class improves code readability and allows
        easy experimentation with hyperparameters.
    """
    data_dir: Path = Path("./data")
    output_dir: Path = Path("./outputs")
    batch_size: int = 128
    z_dim: int = 100
    lr: float = 2e-4
    betas: tuple = (0.5, 0.999)
    epochs: int = 50
    img_size: int = 28
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_interval: int = 300


# ---------------------------------------
#               Model Definitions
# ---------------------------------------

class Generator(nn.Module):
    """
    Generator Network of the GAN.

    The Generator learns to map a random noise vector (latent space)
    to a realistic image resembling the training data distribution.

    Architecture:
        Input (z_dim) → Fully Connected Layers → Output Image

    Layer Details:
        - Linear layers progressively increase dimensionality
        - BatchNorm stabilizes training and accelerates convergence
        - ReLU activation introduces non-linearity
        - Final Tanh activation outputs values in range [-1, 1]

    Args:
        z_dim (int): Dimension of input noise vector
        img_dim (int): Flattened image size (e.g., 28*28 = 784)

    Forward Flow:
        Noise (z) → Neural Network → Fake Image

    Goal:
        Generate images that are indistinguishable from real data,
        effectively "fooling" the Discriminator.
    """
    def __init__(self, z_dim: int, img_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(True),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(True),
            nn.Linear(1024, img_dim), nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class Discriminator(nn.Module):
    """
    Discriminator Network of the GAN.

    The Discriminator acts as a binary classifier that distinguishes
    between real images (from dataset) and fake images (from Generator).

    Architecture:
        Input Image → Fully Connected Layers → Single Output

    Layer Details:
        - Linear layers reduce dimensionality
        - LeakyReLU prevents dying neuron problem
        - Final output is a single logit (real vs fake score)

    Args:
        img_dim (int): Flattened image size

    Forward Flow:
        Image → Neural Network → Real/Fake Score

    Output:
        A scalar value (logit) representing the probability of the image being real.

    Note:
        BCEWithLogitsLoss is used, so no sigmoid activation is applied here.
    """
    def __init__(self, img_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 512), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.model(img)


# ---------------------------------------
#               Data Loader
# ---------------------------------------

def get_dataloader(cfg: GANConfig) -> DataLoader:
    """
    Creates and returns a DataLoader for the MNIST dataset.

    Preprocessing Steps:
        - Resize images to desired size
        - Convert images to tensor format
        - Normalize pixel values from [0,1] → [-1,1]

    Args:
        cfg (GANConfig): Configuration object containing data settings

    Returns:
        DataLoader: Iterable over the dataset

    Note:
        Normalization to [-1,1] is important because the Generator
        uses Tanh activation, which outputs values in the same range.
    """
    transform = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.MNIST(cfg.data_dir, download=True, train=True, transform=transform)

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )


# ---------------------------------------
#               Training Loop
# ---------------------------------------

def train(cfg: GANConfig):
    """
    Trains a Generative Adversarial Network (GAN) on the MNIST dataset.

    Training Strategy:
        GAN training consists of two competing networks:
        1. Discriminator → learns to distinguish real vs fake images
        2. Generator → learns to produce realistic fake images

    Training Loop Overview:
        For each batch:
            1. Train Discriminator:
                - Maximize ability to classify real and fake images correctly
            2. Train Generator:
                - Minimize ability of Discriminator to detect fake images

    Steps:
        - Initialize models and apply weight initialization
        - Define loss function (BCEWithLogitsLoss)
        - Create optimizers (Adam)
        - Iterate over epochs and batches:
            a) Generate fake images from noise
            b) Compute Discriminator loss on real and fake images
            c) Update Discriminator
            d) Compute Generator loss (fooling Discriminator)
            e) Update Generator
        - Log losses and visualize outputs periodically
        - Save trained model weights

    Key Concepts:
        - Adversarial Learning: Generator vs Discriminator
        - Latent Space: Random noise input to Generator
        - Detach: Prevent gradients flowing into Generator during D update

    Outputs:
        - Trained Generator and Discriminator models
        - TensorBoard logs
        - Periodic visualizations of generated images

    Note:
        GAN training is unstable; monitoring both losses and visual output
        is essential for evaluating performance.
    """

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(cfg.output_dir / "logs"))
    dataloader = get_dataloader(cfg)
    img_dim = cfg.img_size * cfg.img_size

    gen = Generator(cfg.z_dim, img_dim).to(cfg.device)
    disc = Discriminator(img_dim).to(cfg.device)

    gen.apply(weights_init)
    disc.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()

    opt_gen = optim.Adam(gen.parameters(), lr=cfg.lr, betas=cfg.betas)
    opt_disc = optim.Adam(disc.parameters(), lr=cfg.lr, betas=cfg.betas)

    step = 0

    for epoch in range(cfg.epochs):
        for real_imgs, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}"):

            bsz = real_imgs.size(0)
            real = real_imgs.view(bsz, -1).to(cfg.device)

            # ---- Train Discriminator ----
            opt_disc.zero_grad()

            noise = torch.randn(bsz, cfg.z_dim, device=cfg.device)
            fake = gen(noise)

            real_labels = torch.ones(bsz, 1, device=cfg.device)
            fake_labels = torch.zeros(bsz, 1, device=cfg.device)

            loss_disc = (
                criterion(disc(real), real_labels) +
                criterion(disc(fake.detach()), fake_labels)
            ) / 2

            loss_disc.backward()
            opt_disc.step()

            # ---- Train Generator ----
            opt_gen.zero_grad()

            output = disc(fake)
            loss_gen = criterion(output, real_labels)

            loss_gen.backward()
            opt_gen.step()

            # ---- Logging ----
            if step % cfg.log_interval == 0:
                writer.add_scalar("Loss/Generator", loss_gen.item(), step)
                writer.add_scalar("Loss/Discriminator", loss_disc.item(), step)

            step += 1

    # Save models
    writer.close()
    torch.save(gen.state_dict(), cfg.output_dir / "generator.pth")
    torch.save(disc.state_dict(), cfg.output_dir / "discriminator.pth")


# ---------------------------------------
#                 Main
# ---------------------------------------

if __name__ == '__main__':
    """
    Entry point for GAN training.

    Allows command-line customization of parameters.
    """

    parser = argparse.ArgumentParser(description="Train GAN on MNIST")

    parser.add_argument('--data_dir', type=Path, default=GANConfig.data_dir)
    parser.add_argument('--output_dir', type=Path, default=GANConfig.output_dir)
    parser.add_argument('--batch_size', type=int, default=GANConfig.batch_size)
    parser.add_argument('--z_dim', type=int, default=GANConfig.z_dim)
    parser.add_argument('--lr', type=float, default=GANConfig.lr)
    parser.add_argument('--epochs', type=int, default=GANConfig.epochs)

    args, _ = parser.parse_known_args()

    config = GANConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        z_dim=args.z_dim,
        lr=args.lr,
        epochs=args.epochs
    )

    train(config)