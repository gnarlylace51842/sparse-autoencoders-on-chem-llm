"""Step 1: load SMILES and extract ChemBERTa hidden states from one layer."""

from __future__ import annotations

import argparse

from chemsae.activations import activations_dir_for_layer, extract_activations
from chemsae.config import CFG, ensure_dirs
from chemsae.data import load_smiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=CFG.activation.layer)
    parser.add_argument("--n", type=int, default=CFG.data.n_molecules)
    parser.add_argument("--batch-size", type=int, default=CFG.activation.batch_size)
    args = parser.parse_args()

    ensure_dirs()
    smiles = load_smiles(n=args.n)
    print(f"loaded {len(smiles)} canonical SMILES")

    bundle = extract_activations(smiles, layer=args.layer, batch_size=args.batch_size)
    out_dir = activations_dir_for_layer(args.layer)
    bundle.save(out_dir)
    print(
        f"saved layer {args.layer} activations to {out_dir}: "
        f"{bundle.tokens.shape[0]} tokens, {len(bundle.smiles)} molecules"
    )


if __name__ == "__main__":
    main()
