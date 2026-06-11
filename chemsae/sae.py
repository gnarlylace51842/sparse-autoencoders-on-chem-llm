"""Sparse autoencoder model.

Implements the standard L1-sparse formulation from Bricken et al. (2023):

    z     = ReLU(W_enc (x - b_dec) + b_enc)
    x_hat = W_dec z + b_dec
    loss  = MSE(x_hat, x) + l1 * ||z||_1

with unit-norm decoder columns (enforced after every step), tied init, and
running counters for dead-feature detection.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, tied_init: bool = True) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden

        W_dec = torch.randn(d_in, d_hidden)
        W_dec = W_dec / W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.W_dec = nn.Parameter(W_dec)
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        if tied_init:
            W_enc = W_dec.T.clone()
        else:
            W_enc = torch.randn(d_hidden, d_in) * (1.0 / d_in**0.5)
        self.W_enc = nn.Parameter(W_enc)
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))

        # n_steps since each feature last fired above threshold; for dead-feature stats.
        self.register_buffer("steps_since_active", torch.zeros(d_hidden, dtype=torch.long))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec.T + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def loss(
        self, x: torch.Tensor, l1: float
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x_hat, z = self(x)
        mse = (x_hat - x).pow(2).mean()
        l1_term = z.abs().sum(dim=-1).mean()
        total = mse + l1 * l1_term

        with torch.no_grad():
            active = (z > 1e-8).any(dim=0)
            self.steps_since_active += 1
            self.steps_since_active[active] = 0
            frac_active = active.float().mean()
            l0 = (z > 1e-8).float().sum(dim=-1).mean()

        metrics = {
            "loss": total.detach(),
            "mse": mse.detach(),
            "l1": l1_term.detach(),
            "frac_active": frac_active,
            "l0": l0,
        }
        return total, metrics

    @torch.no_grad()
    def renormalize_decoder(self) -> None:
        """Unit-norm decoder columns. Call after every optimizer step."""
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.W_dec.div_(norms)

    @torch.no_grad()
    def remove_parallel_gradient(self) -> None:
        """Project out the component of grad(W_dec) parallel to W_dec.

        Keeps the unit-norm constraint approximately satisfied across the step,
        so the renormalize call only corrects for numerical drift rather than
        actively shrinking decoder columns toward zero.
        """
        if self.W_dec.grad is None:
            return
        g = self.W_dec.grad
        dot = (g * self.W_dec).sum(dim=0, keepdim=True)
        g.sub_(dot * self.W_dec)

    @torch.no_grad()
    def dead_feature_mask(self, dead_threshold_steps: int) -> torch.Tensor:
        return self.steps_since_active >= dead_threshold_steps
