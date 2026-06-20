"""Falsification test: does feature selectivity generalize to unseen molecules?

Every lift number in the main paper is measured in-sample — the SAE is trained
on the same 50k molecules the lift is then computed on. That risks rewarding
in-sample overfitting rather than a feature that genuinely tracks a chemical
concept. Here we split molecules into train/test, train the SAE on train tokens
ONLY, and measure feature lift on held-out test molecules the SAE never saw.

If the high-lift feature count collapses on held-out molecules, the central
claim does not generalize and must be downgraded. If it largely holds, the
claim survives a test that could have failed.
"""

from __future__ import annotations

import numpy as np

from chemsae.activations import ActivationBundle, activations_dir_for_layer
from chemsae.analysis import label_features, molecule_level_max
from chemsae.config import CHECKPOINTS_DIR
from chemsae.training import train_sae

SPLIT = 40000   # molecules [0, SPLIT) train ; [SPLIT, 50000) held-out test


def lift10_count(z_mol, smiles):
    labels = label_features(z_mol, smiles, n_features_to_label=200)
    lifts = [lab.best_lift for lab in labels if lab.best_concept is not None]
    n10 = sum(1 for x in lifts if x >= 10)
    n5 = sum(1 for x in lifts if x >= 5)
    return n10, n5, (max(lifts) if lifts else 0.0)


def molstack(bundle, tok_mask, mol_idx):
    sub_idx = mol_idx[tok_mask]
    uniq = np.unique(sub_idx)
    remap = {m: i for i, m in enumerate(uniq)}
    remapped = np.array([remap[m] for m in sub_idx])
    smiles = [bundle.smiles[m] for m in uniq]
    return bundle.tokens[tok_mask], remapped, smiles, len(uniq)


def main():
    bundle = ActivationBundle.load(activations_dir_for_layer(6))
    mol_idx = bundle.token_meta["molecule_idx"].to_numpy()
    train_mask = mol_idx < SPLIT
    test_mask = ~train_mask
    print(f"train tokens: {int(train_mask.sum())}  test tokens: {int(test_mask.sum())}")

    # train SAE on TRAIN tokens only
    train_tokens = bundle.tokens[train_mask]
    sae = train_sae(
        train_tokens, out_dir=CHECKPOINTS_DIR / "heldout",
        n_steps=30000, tag="heldout", seed=0,
    )

    # evaluate on held-out TEST molecules (never seen in training)
    te_tok, te_idx, te_smiles, n_te = molstack(bundle, test_mask, mol_idx)
    z_te = molecule_level_max(sae, te_tok, te_idx, n_molecules=n_te)
    te10, te5, te_max = lift10_count(z_te, te_smiles)

    # in-sample comparison on TRAIN molecules
    tr_tok, tr_idx, tr_smiles, n_tr = molstack(bundle, train_mask, mol_idx)
    z_tr = molecule_level_max(sae, tr_tok, tr_idx, n_molecules=n_tr)
    tr10, tr5, tr_max = lift10_count(z_tr, tr_smiles)

    print("\n=== feature selectivity: in-sample vs held-out ===")
    print(f"  TRAIN molecules (in-sample):  lift>=10: {tr10:3d}   lift>=5: {tr5:3d}   max {tr_max:.1f}")
    print(f"  TEST  molecules (held-out) :  lift>=10: {te10:3d}   lift>=5: {te5:3d}   max {te_max:.1f}")
    ratio = te10 / tr10 if tr10 else float("nan")
    print(f"  held-out / in-sample (lift>=10): {ratio:.0%}")
    print("\n  verdict:", "GENERALIZES" if ratio >= 0.6 else "DOES NOT GENERALIZE — downgrade claim")


if __name__ == "__main__":
    main()
