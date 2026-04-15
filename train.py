# ---------------------------------------
#               Imports
# ---------------------------------------
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
    """Initialize model weights according to DCGAN paper guidelines."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def show(tensor: torch.Tensor, ch: int = 1, size: tuple = (28, 28), num: int = 16):
    """
    Detach, reshape, normalize to [0,1], and plot a grid of generated or real images.
    """
    # reshape and move to CPU
    data = tensor.detach().cpu().view(-1, ch, *size)
    # map from [-1,1] to [0,1]
    data = (data + 1) / 2
    data = data.clamp(0, 1)
    # make grid
    grid = utils.make_grid(data[:num], nrow=4)
    # convert CHW to HWC
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
    data_dir: Path = Path("./data")
    output_dir: Path = Path("./outputs")
    batch_size: int = 128
    z_dim: int = 100
    lr: float = 2e-4
    betas: tuple = (0.5, 0.999)
    epochs: int = 50
    img_size: int = 28
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_interval: int = 300  # steps between prints/plots


# ---------------------------------------
#               Model Definitions
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
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class Discriminator(nn.Module):
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
    transform = transforms.Compose([
        transforms.Resize(cfg.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.MNIST(cfg.data_dir, download=True, train=True, transform=transform)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)


# ---------------------------------------
#               Training Loop
# ---------------------------------------

'''Neural networks learn by minimizing error through gradient-based optimization, 
   gradually improving predictions to match real data patterns.'''

def train(cfg: GANConfig):
    # prepare
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(cfg.output_dir / "logs"))
    dataloader = get_dataloader(cfg)
    img_dim = cfg.img_size * cfg.img_size

    # init models
    gen = Generator(cfg.z_dim, img_dim).to(cfg.device)
    disc = Discriminator(img_dim).to(cfg.device)
    gen.apply(weights_init)
    disc.apply(weights_init)

    # optimizers & loss
    criterion = nn.BCEWithLogitsLoss()
    opt_gen = optim.Adam(gen.parameters(), lr=cfg.lr, betas=cfg.betas)
    opt_disc = optim.Adam(disc.parameters(), lr=cfg.lr, betas=cfg.betas)

    # initial preview
    real_batch, labels = next(iter(dataloader))
    print(f"Initial batch shapes - images: {real_batch.shape}, labels: {labels.shape}")
    print(f"First 10 labels: {labels[:10].tolist()}")
    noise0 = torch.randn(cfg.batch_size, cfg.z_dim, device=cfg.device)
    fake0 = gen(noise0[:real_batch.size(0)])
    show(fake0)

    # training
    step = 0
    mean_gen_loss = 0.0
    mean_disc_loss = 0.0
    for epoch in range(cfg.epochs):
        for real_imgs, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}"):
            bsz = real_imgs.size(0)
            real = real_imgs.view(bsz, -1).to(cfg.device)

            # discriminator update
            opt_disc.zero_grad()
            noise = torch.randn(bsz, cfg.z_dim, device=cfg.device)
            fake = gen(noise)
            real_labels = torch.ones(bsz, 1, device=cfg.device)
            fake_labels = torch.zeros(bsz, 1, device=cfg.device)
            disc_real = disc(real)
            disc_fake = disc(fake.detach())
            loss_disc = (criterion(disc_real, real_labels) + criterion(disc_fake, fake_labels)) / 2
            loss_disc.backward()
            opt_disc.step()

            # generator update
            opt_gen.zero_grad()
            output = disc(fake)
            loss_gen = criterion(output, real_labels)
            loss_gen.backward()
            opt_gen.step()

            # accumulate
            mean_disc_loss += loss_disc.item() / cfg.log_interval
            mean_gen_loss += loss_gen.item() / cfg.log_interval

            # logging & visualization
            if step % cfg.log_interval == 0 and step > 0:
                print(f"Epoch {epoch} | Step {step} | Gen loss: {mean_gen_loss:.4f} | Disc loss: {mean_disc_loss:.4f}")
                with torch.no_grad():
                    sample_noise = torch.randn(bsz, cfg.z_dim, device=cfg.device)
                    sample_fake = gen(sample_noise)
                show(sample_fake)
                show(real_imgs.to(cfg.device).view(bsz, 1, cfg.img_size, cfg.img_size))
                mean_gen_loss = 0.0
                mean_disc_loss = 0.0

            if step % cfg.log_interval == 0:
                writer.add_scalar("Loss/Generator", loss_gen.item(), step)
                writer.add_scalar("Loss/Discriminator", loss_disc.item(), step)

            step += 1

    # save models
    writer.close()
    torch.save(gen.state_dict(), cfg.output_dir / "generator.pth")
    torch.save(disc.state_dict(), cfg.output_dir / "discriminator.pth")


# ---------------------------------------
#                 Main
# ---------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a basic GAN on MNIST")
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
