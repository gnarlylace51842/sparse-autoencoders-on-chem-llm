"""Tiny end-to-end pipeline run to verify the stack works.

Loads a small set of SMILES directly (no HuggingFace download required),
extracts activations from layer 6, trains a small SAE for a few hundred
steps, and labels its features. Should complete in a couple of minutes on
M4 Max MPS.
"""

from __future__ import annotations

from chemsae.activations import extract_activations
from chemsae.analysis import encode_bundle_streaming, label_features, labels_to_dataframe
from chemsae.config import CHECKPOINTS_DIR, RESULTS_DIR, ensure_dirs
from chemsae.data import canonicalize_many
from chemsae.training import train_sae


SMOKE_SMILES = [
    # Aromatic / non-aromatic pairs
    "c1ccccc1", "C1CCCCC1", "c1ccc2ccccc2c1", "C1CCC2CCCCC2C1",
    "c1ccncc1", "C1CCNCC1", "Oc1ccccc1", "OC1CCCCC1",
    # Functional groups
    "CC(=O)O", "CCO", "CC(=O)N", "CC(=O)OC",
    "CCN", "CCC", "CC(=O)C", "CCC=O",
    # Halogens
    "Clc1ccccc1", "FC(F)(F)c1ccccc1", "Brc1ccccc1", "Ic1ccccc1",
    # Heterocycles
    "c1ccoc1", "c1ccsc1", "c1cc[nH]c1", "c1ncc[nH]1",
    # Stereo
    "C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O", "CC(N)C(=O)O",
    # Drug-like
    "CN1CCCC1c2cccnc2", "CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "O=C(C)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1",
] * 5  # duplicate to get ~160 molecules


def main() -> None:
    ensure_dirs()
    smiles = canonicalize_many(SMOKE_SMILES)
    smiles = list(dict.fromkeys(smiles))
    print(f"smoke set: {len(smiles)} unique canonical SMILES")

    bundle = extract_activations(smiles, layer=6, batch_size=32)
    print(f"got {bundle.tokens.shape[0]} non-special tokens at layer 6")

    out_dir = CHECKPOINTS_DIR / "smoke"
    sae = train_sae(
        activations=bundle.tokens,
        out_dir=out_dir,
        expansion=4,
        l1_coeff=1e-3,
        n_steps=500,
        warmup_steps=100,
        batch_size=256,
        tag="smoke",
    )

    z_mol = encode_bundle_streaming(sae, bundle)
    labels = label_features(
        z_mol=z_mol,
        smiles=bundle.smiles,
        top_k=5,
        n_features_to_label=64,
        min_precision=0.5,
        min_lift=1.5,
    )
    df = labels_to_dataframe(labels).sort_values("best_lift", ascending=False, na_position="last")
    out_path = RESULTS_DIR / "smoke_labels.csv"
    df.to_csv(out_path, index=False)
    n_labeled = int(df["best_concept"].notna().sum())
    print(f"smoke: {n_labeled}/{len(df)} features matched a concept")
    print(df.head(15).to_string(index=False))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
