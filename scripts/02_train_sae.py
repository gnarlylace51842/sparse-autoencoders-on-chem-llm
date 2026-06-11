"""Step 2: train SAE on cached activations from one layer."""

from __future__ import annotations

import argparse

from chemsae.activations import ActivationBundle, activations_dir_for_layer
from chemsae.config import CFG, CHECKPOINTS_DIR, ensure_dirs
from chemsae.training import train_sae


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=CFG.activation.layer)
    parser.add_argument("--expansion", type=int, default=CFG.sae.expansion)
    parser.add_argument("--l1", type=float, default=CFG.sae.l1_coeff)
    parser.add_argument("--steps", type=int, default=CFG.sae.n_steps)
    parser.add_argument("--lr", type=float, default=CFG.sae.lr)
    parser.add_argument("--batch-size", type=int, default=CFG.sae.batch_size)
    parser.add_argument("--tag", type=str, default="sae")
    parser.add_argument("--seed", type=int, default=CFG.sae.seed)
    args = parser.parse_args()

    ensure_dirs()
    bundle = ActivationBundle.load(activations_dir_for_layer(args.layer))
    print(f"loaded {bundle.tokens.shape[0]} token activations from layer {args.layer}")

    out_dir = CHECKPOINTS_DIR / f"layer_{args.layer:02d}"
    train_sae(
        activations=bundle.tokens,
        out_dir=out_dir,
        expansion=args.expansion,
        l1_coeff=args.l1,
        n_steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        tag=args.tag,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
