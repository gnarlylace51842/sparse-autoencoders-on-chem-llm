"""Localize the stereochemistry negative result: ChemBERTa or the SAE?

scripts/09 showed the SAE does not distinguish enantiomers. That leaves two
possibilities: (i) ChemBERTa itself barely separates a molecule from its
mirror image, so there is nothing for the SAE to recover; or (ii) ChemBERTa
encodes stereochemistry but the sparse autoencoder discards it.

We test this directly in ChemBERTa's *raw* layer-6 activations (no SAE). For
each enantiomer pair we measure the distance between the two mirror-image
molecules' mean-pooled activations, relative to the distance between random
*different* molecules. If enantiomers sit far closer than random molecules,
the stereochemistry signal is weak in ChemBERTa itself.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger

from chemsae.activations import extract_activations
from chemsae.config import DATA_DIR
import pandas as pd

RDLogger.DisableLog("rdApp.*")
N_PAIRS = 60


def enantiomer(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    centers = Chem.FindMolChiralCenters(m, useLegacyImplementation=False, includeUnassigned=False)
    if len(centers) != 1:
        return None
    a = m.GetAtomWithIdx(centers[0][0])
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
        e = enantiomer(smi)
        if e:
            pairs.append((smi, e))
        if len(pairs) >= N_PAIRS:
            break
    flat = [s for p in pairs for s in p]
    bundle = extract_activations(flat, layer=6)
    a = bundle.pooled.numpy()                 # [120, 768] mean-pooled raw activations
    orig, ent = a[0::2], a[1::2]              # [60, 768] each

    d_enant = np.linalg.norm(orig - ent, axis=1)            # mirror-image distance
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(orig))
    # ensure no molecule paired with itself
    perm[perm == np.arange(len(orig))] = (perm[perm == np.arange(len(orig))] + 1) % len(orig)
    d_rand = np.linalg.norm(orig - orig[perm], axis=1)      # random different-molecule distance

    def cos(x, y):
        return (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))

    print("=== raw ChemBERTa layer-6 activations (no SAE) ===")
    print(f"  enantiomer pair distance:      mean {d_enant.mean():.3f}  median {np.median(d_enant):.3f}")
    print(f"  random molecule pair distance: mean {d_rand.mean():.3f}  median {np.median(d_rand):.3f}")
    print(f"  ratio (enantiomer / random):   {d_enant.mean() / d_rand.mean():.3f}")
    print(f"  cosine sim, enantiomer pairs:  mean {cos(orig, ent).mean():.4f}")
    print(f"  cosine sim, random pairs:      mean {cos(orig, orig[perm]).mean():.4f}")
    print()
    if d_enant.mean() / d_rand.mean() < 0.25:
        print("  -> enantiomers are far closer than random molecules:")
        print("     stereochemistry is weakly encoded in ChemBERTa ITSELF;")
        print("     the SAE's chirality-blindness reflects the base model, not the SAE.")
    else:
        print("  -> ChemBERTa separates enantiomers; the SAE discards that signal.")


if __name__ == "__main__":
    main()
