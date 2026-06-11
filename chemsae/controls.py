"""Control experiments: shuffled activations, specificity tests, layer comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .activations import ActivationBundle, extract_activations
from .analysis import molecule_level_max
from .sae import SparseAutoencoder


def shuffle_per_dimension(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Independently permute each feature column across rows.

    Destroys all cross-feature correlation while preserving marginal
    distributions of each activation dimension. SAE trained on this should
    not learn coherent features.
    """
    g = torch.Generator().manual_seed(seed)
    out = torch.empty_like(x)
    n = x.shape[0]
    for j in range(x.shape[1]):
        perm = torch.randperm(n, generator=g)
        out[:, j] = x[perm, j]
    return out


# Curated paired SMILES — each pair holds a concept fixed while varying one
# property. Positive example matches the concept; negative does not.
SPECIFICITY_PAIRS: list[dict[str, str]] = [
    {"concept": "aromatic_ring_any", "pos": "c1ccccc1", "neg": "C1CCCCC1", "pos_name": "benzene", "neg_name": "cyclohexane"},
    {"concept": "aromatic_ring_any", "pos": "c1ccc2ccccc2c1", "neg": "C1CCC2CCCCC2C1", "pos_name": "naphthalene", "neg_name": "decalin"},
    {"concept": "pyridine", "pos": "c1ccncc1", "neg": "C1CCNCC1", "pos_name": "pyridine", "neg_name": "piperidine"},
    {"concept": "phenol", "pos": "Oc1ccccc1", "neg": "OC1CCCCC1", "pos_name": "phenol", "neg_name": "cyclohexanol"},
    {"concept": "carboxylic_acid", "pos": "CC(=O)O", "neg": "CCO", "pos_name": "acetic_acid", "neg_name": "ethanol"},
    {"concept": "amine_primary", "pos": "CCN", "neg": "CCC", "pos_name": "ethylamine", "neg_name": "propane"},
    {"concept": "amide", "pos": "CC(=O)N", "neg": "CC(=O)O", "pos_name": "acetamide", "neg_name": "acetic_acid"},
    {"concept": "ester", "pos": "CC(=O)OC", "neg": "CC(=O)O", "pos_name": "methyl_acetate", "neg_name": "acetic_acid"},
    {"concept": "halogen_any", "pos": "Clc1ccccc1", "neg": "c1ccccc1", "pos_name": "chlorobenzene", "neg_name": "benzene"},
    {"concept": "trifluoromethyl", "pos": "FC(F)(F)c1ccccc1", "neg": "Cc1ccccc1", "pos_name": "trifluorotoluene", "neg_name": "toluene"},
    {"concept": "nitro", "pos": "O=[N+]([O-])c1ccccc1", "neg": "c1ccccc1", "pos_name": "nitrobenzene", "neg_name": "benzene"},
    {"concept": "nitrile", "pos": "N#Cc1ccccc1", "neg": "c1ccccc1", "pos_name": "benzonitrile", "neg_name": "benzene"},
    {"concept": "chiral_center", "pos": "C[C@@H](N)C(=O)O", "neg": "CC(N)C(=O)O", "pos_name": "L_alanine", "neg_name": "alanine_achiral"},
    {"concept": "chiral_R_vs_S", "pos": "C[C@H](N)C(=O)O", "neg": "C[C@@H](N)C(=O)O", "pos_name": "D_alanine", "neg_name": "L_alanine"},
    {"concept": "ketone", "pos": "CC(=O)C", "neg": "CC(O)C", "pos_name": "acetone", "neg_name": "isopropanol"},
    {"concept": "alkene", "pos": "C=CC", "neg": "CCC", "pos_name": "propene", "neg_name": "propane"},
    {"concept": "alkyne", "pos": "C#CC", "neg": "CCC", "pos_name": "propyne", "neg_name": "propane"},
    {"concept": "furan", "pos": "c1ccoc1", "neg": "C1CCOC1", "pos_name": "furan", "neg_name": "tetrahydrofuran"},
    {"concept": "thiophene", "pos": "c1ccsc1", "neg": "C1CCSC1", "pos_name": "thiophene", "neg_name": "thiolane"},
]


@dataclass
class SpecificityResult:
    concept: str
    pos_name: str
    neg_name: str
    feature_id: int | None       # the SAE feature labeled with this concept (if any)
    pos_activation: float
    neg_activation: float
    delta: float                 # pos - neg


def run_specificity_tests(
    sae: SparseAutoencoder,
    layer: int,
    feature_concept_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    """For each (pos, neg) pair, extract activations through ChemBERTa+SAE and compare.

    `feature_concept_map` maps a concept name -> feature_id that was labeled as
    that concept in the main run. If provided, we evaluate that specific
    feature's activation on (pos, neg). Otherwise we evaluate the
    most-activated feature across both molecules.
    """
    all_smiles = []
    pair_keys = []
    for pair in SPECIFICITY_PAIRS:
        all_smiles.extend([pair["pos"], pair["neg"]])
        pair_keys.extend([(pair["concept"], "pos"), (pair["concept"], "neg")])

    bundle = extract_activations(all_smiles, layer=layer)
    z_mol = molecule_level_max(
        sae,
        tokens=bundle.tokens,
        molecule_idx=bundle.token_meta["molecule_idx"].to_numpy(),
        n_molecules=len(all_smiles),
    )

    rows = []
    for i, pair in enumerate(SPECIFICITY_PAIRS):
        pos_mol = 2 * i
        neg_mol = 2 * i + 1
        fid = (feature_concept_map or {}).get(pair["concept"])
        if fid is None:
            diff = (z_mol[pos_mol] - z_mol[neg_mol]).abs()
            fid = int(diff.argmax().item())
        pos_act = float(z_mol[pos_mol, fid].item())
        neg_act = float(z_mol[neg_mol, fid].item())
        rows.append(
            {
                "concept": pair["concept"],
                "pos_name": pair["pos_name"],
                "neg_name": pair["neg_name"],
                "feature_id": fid,
                "pos_activation": pos_act,
                "neg_activation": neg_act,
                "delta": pos_act - neg_act,
            }
        )
    return pd.DataFrame(rows)


def summarize_layer_comparison(layer_labels: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Given a dict layer -> labels DataFrame, build a per-layer summary."""
    rows = []
    for layer, df in layer_labels.items():
        labeled = df.dropna(subset=["best_concept"])
        rows.append(
            {
                "layer": layer,
                "n_features_analyzed": len(df),
                "n_labeled": len(labeled),
                "frac_labeled": len(labeled) / max(len(df), 1),
                "median_precision": float(labeled["best_precision"].median()) if len(labeled) else float("nan"),
                "median_lift": float(labeled["best_lift"].median()) if len(labeled) else float("nan"),
                "unique_concepts": labeled["best_concept"].nunique() if len(labeled) else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)
