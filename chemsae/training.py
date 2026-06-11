"""Train a sparse autoencoder on cached ChemBERTa activations."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import trange

from .config import CFG, DEVICE, DTYPE
from .sae import SparseAutoencoder


def _l1_schedule(step: int, warmup_steps: int, target: float) -> float:
    if warmup_steps <= 0:
        return target
    return target * min(1.0, step / warmup_steps)


def _sample_batch(activations: torch.Tensor, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    n = activations.shape[0]
    idx = torch.randint(0, n, (batch_size,), generator=generator)
    return activations[idx]


def train_sae(
    activations: torch.Tensor,
    out_dir: Path,
    expansion: int | None = None,
    l1_coeff: float | None = None,
    n_steps: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    warmup_steps: int | None = None,
    seed: int | None = None,
    tag: str = "sae",
) -> SparseAutoencoder:
    """Train SAE on a [N, d_model] tensor and checkpoint to `out_dir`."""
    cfg = CFG.sae
    expansion = expansion or cfg.expansion
    l1_coeff = l1_coeff or cfg.l1_coeff
    n_steps = n_steps or cfg.n_steps
    lr = lr or cfg.lr
    batch_size = batch_size or cfg.batch_size
    warmup_steps = warmup_steps or cfg.warmup_steps
    seed = cfg.seed if seed is None else seed

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    d_in = activations.shape[1]
    d_hidden = d_in * expansion
    sae = SparseAutoencoder(d_in=d_in, d_hidden=d_hidden, tied_init=cfg.tied_init).to(DEVICE).to(DTYPE)

    with torch.no_grad():
        sae.b_dec.copy_(activations.mean(dim=0).to(DEVICE).to(DTYPE))

    opt = Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    activations = activations.to(DTYPE)

    gen = torch.Generator().manual_seed(seed)
    history: list[dict] = []

    t0 = time.time()
    bar = trange(n_steps, desc=f"train {tag}")
    for step in bar:
        x = _sample_batch(activations, batch_size, gen).to(DEVICE)
        l1 = _l1_schedule(step, warmup_steps, l1_coeff)

        loss, metrics = sae.loss(x, l1=l1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sae.remove_parallel_gradient()
        opt.step()
        sae.renormalize_decoder()

        if step % cfg.log_every == 0 or step == n_steps - 1:
            row = {
                "step": step,
                "l1_coeff": l1,
                **{k: float(v) for k, v in metrics.items()},
                "n_dead_2k": int((sae.steps_since_active >= 2000).sum()),
            }
            history.append(row)
            bar.set_postfix(
                mse=f"{row['mse']:.4f}",
                l0=f"{row['l0']:.1f}",
                dead=row["n_dead_2k"],
            )

        if (step + 1) % cfg.save_every == 0:
            _save_checkpoint(sae, out_dir, history, tag=f"{tag}_step{step+1}")

    _save_checkpoint(sae, out_dir, history, tag=f"{tag}_final")
    (out_dir / f"{tag}_history.json").write_text(json.dumps(history, indent=2))
    print(f"trained {tag} in {time.time() - t0:.1f}s")
    return sae


def _save_checkpoint(sae: SparseAutoencoder, out_dir: Path, history: list[dict], tag: str) -> None:
    torch.save(
        {
            "state_dict": sae.state_dict(),
            "d_in": sae.d_in,
            "d_hidden": sae.d_hidden,
            "history_len": len(history),
        },
        out_dir / f"{tag}.pt",
    )


def load_sae(checkpoint_path: Path) -> SparseAutoencoder:
    blob = torch.load(checkpoint_path, map_location=DEVICE)
    sae = SparseAutoencoder(d_in=blob["d_in"], d_hidden=blob["d_hidden"], tied_init=False)
    sae.load_state_dict(blob["state_dict"])
    sae.to(DEVICE).to(DTYPE)
    sae.eval()
    return sae
