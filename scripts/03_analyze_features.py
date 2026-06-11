"""Step 3: encode activations through trained SAE, label features by concept."""

from __future__ import annotations

import argparse

import torch

from chemsae.activations import ActivationBundle, activations_dir_for_layer
from chemsae.analysis import (
    encode_bundle_streaming,
    label_features,
    save_labels,
    save_token_level,
    top_tokens_per_feature,
)
from chemsae.config import CFG, CHECKPOINTS_DIR, RESULTS_DIR, ensure_dirs
from chemsae.training import load_sae


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=CFG.activation.layer)
    parser.add_argument("--tag", type=str, default="sae")
    parser.add_argument("--top-k", type=int, default=CFG.analysis.top_k)
    parser.add_argument("--n-features", type=int, default=CFG.analysis.n_features_to_label)
    args = parser.parse_args()

    ensure_dirs()
    ckpt = CHECKPOINTS_DIR / f"layer_{args.layer:02d}" / f"{args.tag}_final.pt"
    sae = load_sae(ckpt)
    bundle = ActivationBundle.load(activations_dir_for_layer(args.layer))

    print("encoding activations through SAE (streaming)…")
    z_mol = encode_bundle_streaming(sae, bundle)
    torch.save(z_mol, RESULTS_DIR / f"z_mol_layer{args.layer:02d}_{args.tag}.pt")

    print("labeling features…")
    labels = label_features(
        z_mol=z_mol,
        smiles=bundle.smiles,
        top_k=args.top_k,
        n_features_to_label=args.n_features,
    )
    out_dir = RESULTS_DIR / f"layer_{args.layer:02d}"
    save_labels(labels, out_dir, tag=args.tag)
    n_labeled = sum(1 for lab in labels if lab.best_concept is not None)
    print(f"analyzed {len(labels)} active features, {n_labeled} matched a curated concept")

    labeled_ids = [lab.feature_id for lab in labels if lab.best_concept is not None]
    if labeled_ids:
        print(f"extracting top tokens for {len(labeled_ids)} labeled features…")
        tlevel = top_tokens_per_feature(
            sae=sae,
            tokens=bundle.tokens,
            token_meta=bundle.token_meta,
            smiles=bundle.smiles,
            feature_ids=labeled_ids,
            k=10,
        )
        save_token_level(tlevel, out_dir, tag=args.tag)


if __name__ == "__main__":
    main()
