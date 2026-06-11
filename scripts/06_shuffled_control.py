"""Step 6: shuffled-activations control.

Train an SAE on activations whose feature dimensions have been independently
permuted across rows (destroys all cross-feature correlation while preserving
marginals). Compare the interpretability of the resulting features to the
real SAE — if our real findings are genuine and not SAE artifacts, the
shuffled SAE should label far fewer features as named chemical concepts.
"""

from __future__ import annotations

import argparse

import pandas as pd

from chemsae.activations import ActivationBundle, activations_dir_for_layer
from chemsae.analysis import encode_bundle_streaming, label_features, labels_to_dataframe, save_labels
from chemsae.config import CFG, CHECKPOINTS_DIR, RESULTS_DIR, ensure_dirs
from chemsae.controls import shuffle_per_dimension
from chemsae.training import train_sae


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=CFG.activation.layer)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ensure_dirs()
    bundle = ActivationBundle.load(activations_dir_for_layer(args.layer))

    print("shuffling activations per dimension…")
    shuffled = shuffle_per_dimension(bundle.tokens, seed=args.seed)

    out_dir = CHECKPOINTS_DIR / f"layer_{args.layer:02d}"
    sae = train_sae(activations=shuffled, out_dir=out_dir, tag="sae_shuffled")

    # For analysis, encode the shuffled activations themselves to find features,
    # then map back to the *original* SMILES via the original token_meta so we
    # can ask whether those "features" correspond to any chemical concepts.
    shuffled_bundle = ActivationBundle(
        tokens=shuffled,
        token_meta=bundle.token_meta,
        pooled=bundle.pooled,
        cls=bundle.cls,
        smiles=bundle.smiles,
        layer=bundle.layer,
    )
    z_mol = encode_bundle_streaming(sae, shuffled_bundle)
    labels = label_features(z_mol=z_mol, smiles=bundle.smiles)
    out = RESULTS_DIR / f"layer_{args.layer:02d}"
    save_labels(labels, out, tag="sae_shuffled")

    df = labels_to_dataframe(labels)
    n_labeled = int(df["best_concept"].notna().sum())
    print(f"shuffled control: {n_labeled} / {len(df)} features matched a concept")


if __name__ == "__main__":
    main()
