"""Out-of-sample test of whether the SAE encodes stereochemistry.

Feature 2486 was found post hoc to separate one D/L-alanine pair. That is a
garden-of-forking-paths result (one feature chosen from 3,072 on one pair).
Here we test it properly: build many *in-distribution* single-stereocenter
enantiomer pairs from the training corpus, and ask whether feature 2486 (and
features in general) reliably discriminate enantiomers it was never selected on.

A genuine "stereochemistry feature" should activate differently on a molecule
and its mirror image across many independent pairs, not just one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

from chemsae.activations import extract_activations
from chemsae.analysis import molecule_level_max
from chemsae.config import CHECKPOINTS_DIR, DATA_DIR
from chemsae.training import load_sae

RDLogger.DisableLog("rdApp.*")
CANDIDATE = 2486          # the post-hoc feature from the alanine pair
N_PAIRS = 60


def enantiomer(smi: str) -> str | None:
    """Return the mirror image of a single-stereocenter molecule, else None."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    centers = Chem.FindMolChiralCenters(m, useLegacyImplementation=False, includeUnassigned=False)
    if len(centers) != 1:
        return None
    idx = centers[0][0]
    a = m.GetAtomWithIdx(idx)
    tag = a.GetChiralTag()
    if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
        a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
        a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    else:
        return None
    out = Chem.MolToSmiles(m)
    return out if out != smi else None


def main():
    smiles = pd.read_parquet(DATA_DIR / "smiles.parquet")["smiles"].tolist()
    pairs = []
    for smi in smiles:
        ent = enantiomer(smi)
        if ent:
            pairs.append((smi, ent))
        if len(pairs) >= N_PAIRS:
            break
    print(f"built {len(pairs)} single-stereocenter enantiomer pairs (in-distribution)")

    flat = [s for p in pairs for s in p]               # orig0, ent0, orig1, ent1, ...
    bundle = extract_activations(flat, layer=6)
    sae = load_sae(CHECKPOINTS_DIR / "layer_06" / "sae_final.pt")
    z = molecule_level_max(
        sae, bundle.tokens,
        bundle.token_meta["molecule_idx"].to_numpy(), n_molecules=len(flat),
    ).numpy()

    orig = z[0::2]                                      # [N, 3072]
    ent = z[1::2]
    diff = orig - ent                                   # within-pair difference

    # --- pre-registered candidate feature 2486 ---
    d = diff[:, CANDIDATE]
    sep = np.abs(d) > 1.0
    print(f"\n=== candidate feature {CANDIDATE} (found on alanine) ===")
    print(f"  mean |activation difference| across {len(pairs)} new pairs: {np.abs(d).mean():.3f}")
    print(f"  pairs it separates (|diff|>1): {sep.sum()}/{len(pairs)} ({sep.mean():.0%})")
    print(f"  sign consistency (same direction): {max((d>0).mean(),(d<0).mean()):.0%}")
    print(f"  activation on originals: mean {orig[:,CANDIDATE].mean():.2f}, on enantiomers: {ent[:,CANDIDATE].mean():.2f}")

    # --- is ANY feature a consistent stereo-discriminator? ---
    abs_diff = np.abs(diff)
    mean_abs = abs_diff.mean(axis=0)                    # per-feature mean within-pair |diff|
    consistency = np.maximum((diff > 0).mean(axis=0), (diff < 0).mean(axis=0))
    # a real stereo feature: large AND directionally consistent across pairs
    score = mean_abs * (consistency >= 0.8)
    top = np.argsort(-score)[:8]
    print(f"\n=== top features by consistent within-pair discrimination ===")
    print(f"  (of 3072 features; consistency = same sign on >=80% of pairs)")
    for f in top:
        print(f"  feat {f:4d}: mean|diff|={mean_abs[f]:.2f}  sign-consistency={consistency[f]:.0%}  "
              f"separates {int((abs_diff[:, f] > 1.0).sum())}/{len(pairs)}")

    # context: how different are enantiomers overall?
    frac_features_react = (mean_abs > 0.5).mean()
    print(f"\n  fraction of all 3072 features with mean|diff|>0.5: {frac_features_react:.1%}")
    print(f"  median within-pair |diff| over all features: {np.median(mean_abs):.4f}")


if __name__ == "__main__":
    main()
