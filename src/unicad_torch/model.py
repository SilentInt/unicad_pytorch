from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from unicad_torch.config import GalaxyConfig


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


def pretrain_autoencoder(
    model: Autoencoder,
    X: torch.Tensor,
    config: GalaxyConfig,
) -> Autoencoder:
    """Pretrain autoencoder with MSE(sum) loss, Adam + StepLR."""
    model.to(config.device)
    model.train()

    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=config.pretrain_batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.pretrain_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    log_interval = max(1, config.pretrain_epochs // 10)

    for epoch in range(config.pretrain_epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(config.device)
            _, x_hat = model(batch)
            loss = F.mse_loss(x_hat, batch, reduction="sum")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        if config.verbose and (
            epoch % log_interval == 0 or epoch == config.pretrain_epochs - 1
        ):
            print(
                f"  [Pretrain] epoch {epoch + 1}/{config.pretrain_epochs}  "
                f"loss={epoch_loss:.4f}"
            )

    model.eval()
    return model
