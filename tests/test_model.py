"""Tests for Autoencoder and pretrain_autoencoder."""

from __future__ import annotations

import torch

from unicad_torch.config import GalaxyConfig
from unicad_torch.model import Autoencoder, pretrain_autoencoder


class TestAutoencoder:
    def test_forward_shapes(self):
        model = Autoencoder(input_dim=6, hidden_dim=16, latent_dim=8)
        X = torch.randn(10, 6)
        z, x_hat = model(X)
        assert z.shape == (10, 8)
        assert x_hat.shape == (10, 6)

    def test_encoder_separate(self):
        model = Autoencoder(input_dim=6, hidden_dim=16, latent_dim=8)
        X = torch.randn(10, 6)
        z = model.encoder(X)
        assert z.shape == (10, 8)

    def test_latent_equals_hidden(self):
        model = Autoencoder(input_dim=6, hidden_dim=16, latent_dim=16)
        X = torch.randn(10, 6)
        z, x_hat = model(X)
        assert z.shape == (10, 16)

    def test_backward_pass(self):
        model = Autoencoder(input_dim=6, hidden_dim=16, latent_dim=8)
        X = torch.randn(10, 6)
        z, x_hat = model(X)
        loss = ((x_hat - X) ** 2).sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


class TestPretrainAutoencoder:
    def test_pretrain_reduces_loss(self):
        config = GalaxyConfig(pretrain_epochs=10, pretrain_lr=1e-3, seed=42)
        model = Autoencoder(input_dim=6, hidden_dim=16, latent_dim=8)
        X = torch.randn(50, 6)

        with torch.no_grad():
            _, x_hat_before = model(X)
            loss_before = ((x_hat_before - X) ** 2).sum().item()

        pretrain_autoencoder(model, X, config)

        with torch.no_grad():
            _, x_hat_after = model(X)
            loss_after = ((x_hat_after - X) ** 2).sum().item()

        assert loss_after < loss_before

    def test_pretrain_model_eval(self):
        config = GalaxyConfig(pretrain_epochs=2)
        model = Autoencoder(input_dim=6, hidden_dim=16, latent_dim=8)
        X = torch.randn(50, 6)
        pretrain_autoencoder(model, X, config)
        assert not model.training
