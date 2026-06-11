"""Step 5: layer comparison.

Runs extract -> train -> analyze for layers 2, 6, 11 (early / middle / late),
then summarises how many features are interpretable as named concepts at each
depth. Re-uses cached activations and checkpoints where available.
"""

from __future__ import annotations

import argparse

import pandas as pd

from chemsae.activations import ActivationBundle, activations_dir_for_layer, extract_activations
from chemsae.analysis import encode_bundle_streaming, label_features, labels_to_dataframe, save_labels
from chemsae.config import CFG, CHECKPOINTS_DIR, RESULTS_DIR, ensure_dirs
from chemsae.controls import summarize_layer_comparison
from chemsae.data import load_smiles
from chemsae.training import load_sae, train_sae


def _prepare_activations(layer: int) -> ActivationBundle:
    out_dir = activations_dir_for_layer(layer)
    if (out_dir / "tokens.pt").exists():
        print(f"layer {layer}: loading cached activations")
        return ActivationBundle.load(out_dir)
    smiles = load_smiles()
    print(f"layer {layer}: extracting fresh activations")
    bundle = extract_activations(smiles, layer=layer)
    bundle.save(out_dir)
    return bundle


def _prepare_sae(layer: int, bundle: ActivationBundle):
    ckpt_dir = CHECKPOINTS_DIR / f"layer_{layer:02d}"
    final = ckpt_dir / "sae_final.pt"
    if final.exists():
        print(f"layer {layer}: loading trained SAE")
        return load_sae(final)
    print(f"layer {layer}: training SAE")
    return train_sae(activations=bundle.tokens, out_dir=ckpt_dir, tag="sae")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 6, 11])
    args = parser.parse_args()

    ensure_dirs()
    layer_labels: dict[int, pd.DataFrame] = {}
    for layer in args.layers:
        bundle = _prepare_activations(layer)
        sae = _prepare_sae(layer, bundle)
        z_mol = encode_bundle_streaming(sae, bundle)
        labels = label_features(z_mol=z_mol, smiles=bundle.smiles, n_features_to_label=200)
        save_labels(labels, RESULTS_DIR / f"layer_{layer:02d}", tag="sae")
        layer_labels[layer] = labels_to_dataframe(labels)

    summary = summarize_layer_comparison(layer_labels)
    print(summary.to_string(index=False))
    summary.to_csv(RESULTS_DIR / "layer_comparison_summary.csv", index=False)


if __name__ == "__main__":
    main()
