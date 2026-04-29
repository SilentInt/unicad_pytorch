from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from unicad_torch.config import GalaxyConfig

if TYPE_CHECKING:
    from unicad_torch.callbacks import CallbackManager, FitContext

logger = logging.getLogger("unicad_torch.model")


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
    callbacks: CallbackManager | None = None,
    ctx: FitContext | None = None,
) -> Autoencoder:
    """Pretrain autoencoder with MSE(sum) loss, Adam + StepLR.

    If *callbacks* and *ctx* are provided, fires
    ``on_pretrain_epoch_begin`` / ``on_pretrain_epoch_end`` at each
    epoch boundary and checks ``ctx.stop_training``.
    """
    model.to(config.device)
    model.train()

    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=config.pretrain_batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.pretrain_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    for epoch in range(config.pretrain_epochs):
        # --- callback: epoch begin ---
        if ctx is not None:
            ctx.epoch = epoch
            ctx.stage = "pretrain"
            ctx.total_epochs = config.pretrain_epochs
        if callbacks is not None and ctx is not None:
            callbacks.fire("on_pretrain_epoch_begin", ctx)

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

        # --- callback: epoch end ---
        if ctx is not None:
            ctx.pretrain_loss = epoch_loss
        if callbacks is not None and ctx is not None:
            callbacks.fire("on_pretrain_epoch_end", ctx)

        logger.info(
            "[Pretrain] epoch %d/%d  loss=%.4f",
            epoch + 1,
            config.pretrain_epochs,
            epoch_loss,
        )

        if ctx is not None and ctx.stop_training:
            logger.info("Pretraining stopped early at epoch %d", epoch + 1)
            break

    model.eval()
    return model
