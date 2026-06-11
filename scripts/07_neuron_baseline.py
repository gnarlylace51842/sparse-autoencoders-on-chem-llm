"""Neuron baseline: treat each raw ChemBERTa neuron as a candidate feature.

For each of the d_model raw neurons at the chosen layer, aggregate token
activations to molecule level (max over tokens), then run the same concept
labeling pipeline. The comparison vs SAE features answers: did the SAE buy us
anything, or were the neurons already interpretable?
"""

from __future__ import annotations

import argparse

from chemsae.activations import ActivationBundle, activations_dir_for_layer
from chemsae.analysis import aggregate_token_to_molecule, label_features, labels_to_dataframe, save_labels
from chemsae.config import CFG, RESULTS_DIR, ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=CFG.activation.layer)
    parser.add_argument("--n-features", type=int, default=None,
                        help="default: all neurons in the layer")
    args = parser.parse_args()

    ensure_dirs()
    bundle = ActivationBundle.load(activations_dir_for_layer(args.layer))
    tokens = bundle.tokens
    mol_idx = bundle.token_meta["molecule_idx"].to_numpy()
    n_mol = len(bundle.smiles)

    # Take positive part to match SAE convention (ReLU'd features).
    pos = tokens.clamp_min(0.0)
    neuron_per_mol = aggregate_token_to_molecule(pos, mol_idx, n_molecules=n_mol, reduce="max")

    n_features = args.n_features or neuron_per_mol.shape[1]
    labels = label_features(
        z_mol=neuron_per_mol,
        smiles=bundle.smiles,
        n_features_to_label=n_features,
    )
    out = RESULTS_DIR / f"layer_{args.layer:02d}"
    save_labels(labels, out, tag="neurons")

    df = labels_to_dataframe(labels)
    n_labeled = int(df["best_concept"].notna().sum())
    print(f"neuron baseline @ layer {args.layer}: {n_labeled} / {len(df)} neurons matched a concept")


if __name__ == "__main__":
    main()
