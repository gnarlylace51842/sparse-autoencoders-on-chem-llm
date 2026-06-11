"""Feature analysis: top-K activating molecules and concept labeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFMCS
from tqdm import tqdm

from .activations import ActivationBundle
from .concepts import compiled_smarts, match_concepts
from .config import CFG
from .sae import SparseAutoencoder


@torch.no_grad()
def encode_in_chunks(
    sae: SparseAutoencoder, x: torch.Tensor, chunk: int = 16_384
) -> torch.Tensor:
    """Run encoder on `x` in CPU-friendly chunks, return [N, d_hidden] on CPU.

    This materializes the full encoded tensor; for large `x`, prefer
    `molecule_level_max` or `top_tokens_for_features` which stream and avoid
    holding the 4×-wider encoding in memory.
    """
    sae.eval()
    out = torch.empty(x.shape[0], sae.d_hidden, dtype=x.dtype)
    device = next(sae.parameters()).device
    for start in range(0, x.shape[0], chunk):
        batch = x[start : start + chunk].to(device)
        out[start : start + chunk] = sae.encode(batch).cpu()
    return out


@torch.no_grad()
def molecule_level_max(
    sae: SparseAutoencoder,
    tokens: torch.Tensor,
    molecule_idx: np.ndarray,
    n_molecules: int,
    chunk: int = 16_384,
) -> torch.Tensor:
    """Stream encode -> per-molecule max-over-tokens of feature activations.

    Avoids materializing the full [N_tokens, d_hidden] encoded matrix; only
    [chunk, d_hidden] is ever in CPU memory at once on top of the [N_mol,
    d_hidden] output. For d_hidden ≫ d_in (overcomplete SAE) this can save
    >10× memory vs. encode-then-aggregate.
    """
    sae.eval()
    device = next(sae.parameters()).device
    out = torch.full((n_molecules, sae.d_hidden), float("-inf"), dtype=tokens.dtype)
    mol_idx_t = torch.from_numpy(molecule_idx.astype(np.int64))
    for start in range(0, tokens.shape[0], chunk):
        end = start + chunk
        x = tokens[start:end].to(device)
        z = sae.encode(x).cpu()
        idx_chunk = mol_idx_t[start:end].unsqueeze(1).expand_as(z)
        out.scatter_reduce_(0, idx_chunk, z, reduce="amax")
    out[out == float("-inf")] = 0.0
    return out


@torch.no_grad()
def top_tokens_for_features(
    sae: SparseAutoencoder,
    tokens: torch.Tensor,
    feature_ids: list[int],
    k: int = 10,
    chunk: int = 16_384,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For a subset of features, stream-encode tokens and return top-K (vals, idx).

    Returns (top_values, top_token_indices), each shape [len(feature_ids), k].
    """
    sae.eval()
    device = next(sae.parameters()).device
    fids = torch.tensor(feature_ids, dtype=torch.long)
    z_selected = torch.empty(tokens.shape[0], len(feature_ids), dtype=tokens.dtype)
    for start in range(0, tokens.shape[0], chunk):
        end = start + chunk
        x = tokens[start:end].to(device)
        z = sae.encode(x).cpu()
        z_selected[start:end] = z.index_select(1, fids)
    top_vals, top_idx = torch.topk(z_selected, k=k, dim=0)  # [k, n_feat]
    return top_vals.T.contiguous(), top_idx.T.contiguous()


def aggregate_token_to_molecule(
    z_tokens: torch.Tensor, molecule_idx: np.ndarray, n_molecules: int, reduce: str = "max"
) -> torch.Tensor:
    """[N_tokens, d_hidden] -> [N_molecules, d_hidden] via per-molecule reduction."""
    d_hidden = z_tokens.shape[1]
    out = torch.zeros(n_molecules, d_hidden, dtype=z_tokens.dtype)
    counts = torch.zeros(n_molecules, dtype=torch.long)
    mol_idx_t = torch.from_numpy(molecule_idx.astype(np.int64))

    if reduce == "max":
        out.fill_(float("-inf"))
        out.scatter_reduce_(0, mol_idx_t.unsqueeze(1).expand_as(z_tokens), z_tokens, reduce="amax")
        out[out == float("-inf")] = 0.0
    elif reduce == "mean":
        out.scatter_add_(0, mol_idx_t.unsqueeze(1).expand_as(z_tokens), z_tokens)
        counts.scatter_add_(0, mol_idx_t, torch.ones_like(mol_idx_t))
        counts = counts.clamp_min(1).unsqueeze(1).to(z_tokens.dtype)
        out = out / counts
    else:
        raise ValueError(f"unknown reduce: {reduce}")
    return out


def top_k_per_feature(z: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """For each column (feature), return (top-K indices, top-K values). z: [N, d_hidden]."""
    vals, idx = torch.topk(z, k=k, dim=0)  # [k, d_hidden]
    return idx.T.contiguous(), vals.T.contiguous()  # [d_hidden, k]


@dataclass
class FeatureLabel:
    feature_id: int
    n_active: int
    best_concept: str | None
    best_precision: float
    best_lift: float
    mcs_smarts: str | None
    mcs_size: int
    top_smiles: list[str]
    top_values: list[float]
    concept_precisions: dict[str, float]


def label_features(
    z_mol: torch.Tensor,
    smiles: list[str],
    top_k: int | None = None,
    n_features_to_label: int | None = None,
    min_activation: float | None = None,
    mcs_timeout: int | None = None,
    min_precision: float = 0.7,
    min_lift: float = 1.3,
    precision_override: float = 0.95,
) -> list[FeatureLabel]:
    """For each SAE feature, find top-K molecules and label by concept precision/lift."""
    cfg = CFG.analysis
    top_k = top_k or cfg.top_k
    n_features_to_label = n_features_to_label or cfg.n_features_to_label
    min_activation = min_activation if min_activation is not None else cfg.min_activation
    mcs_timeout = mcs_timeout or cfg.mcs_timeout

    n_mol, d_hidden = z_mol.shape
    patterns = compiled_smarts()
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    valid = [m is not None for m in mols]

    baseline = {}
    for name, pat in patterns.items():
        baseline[name] = float(np.mean([m.HasSubstructMatch(pat) for m, v in zip(mols, valid) if v]))

    feature_max = z_mol.max(dim=0).values
    active_features = (feature_max > min_activation).nonzero(as_tuple=True)[0].tolist()
    sorted_features = sorted(active_features, key=lambda i: -float(feature_max[i]))
    selected = sorted_features[:n_features_to_label]

    top_idx, top_vals = top_k_per_feature(z_mol, k=top_k)

    labels: list[FeatureLabel] = []
    for fid in tqdm(selected, desc="labeling features"):
        idxs = top_idx[fid].tolist()
        vals = top_vals[fid].tolist()
        top_mols = [mols[i] for i in idxs if valid[i]]
        top_smi = [smiles[i] for i in idxs if valid[i]]

        if len(top_mols) < max(3, top_k // 4):
            continue

        per_concept_hits = {name: 0 for name in patterns}
        for m in top_mols:
            for name, pat in patterns.items():
                if m.HasSubstructMatch(pat):
                    per_concept_hits[name] += 1
        precisions = {name: per_concept_hits[name] / len(top_mols) for name in patterns}

        # A concept "qualifies" if it's both common in the top-K and selective
        # over the baseline. Very-high-precision (>=precision_override) gets a
        # pass on the lift requirement so concepts with high baseline (e.g.,
        # "aromatic_ring_any") can still be labeled when a feature is clearly
        # about them.
        best_concept = None
        best_precision = 0.0
        best_lift = 0.0
        for name, prec in precisions.items():
            base = max(baseline[name], 1e-6)
            lift = prec / base
            qualifies = (prec >= min_precision and lift >= min_lift) or prec >= precision_override
            if qualifies and lift > best_lift:
                best_concept = name
                best_precision = prec
                best_lift = lift

        mcs_smarts, mcs_size = _safe_mcs(top_mols, mcs_timeout)
        n_active = int((z_mol[:, fid] > min_activation).sum().item())

        labels.append(
            FeatureLabel(
                feature_id=fid,
                n_active=n_active,
                best_concept=best_concept,
                best_precision=best_precision,
                best_lift=best_lift,
                mcs_smarts=mcs_smarts,
                mcs_size=mcs_size,
                top_smiles=top_smi,
                top_values=vals[: len(top_smi)],
                concept_precisions=precisions,
            )
        )
    return labels


def _safe_mcs(mols: list[Chem.Mol], timeout: int) -> tuple[str | None, int]:
    if len(mols) < 2:
        return None, 0
    try:
        res = rdFMCS.FindMCS(
            mols,
            timeout=timeout,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            ringMatchesRingOnly=False,
            completeRingsOnly=False,
        )
        if res.canceled or res.numAtoms < 3:
            return None, int(res.numAtoms)
        return res.smartsString, int(res.numAtoms)
    except Exception:
        return None, 0


def labels_to_dataframe(labels: list[FeatureLabel]) -> pd.DataFrame:
    rows = []
    for lab in labels:
        rows.append(
            {
                "feature_id": lab.feature_id,
                "n_active": lab.n_active,
                "best_concept": lab.best_concept,
                "best_precision": lab.best_precision,
                "best_lift": lab.best_lift,
                "mcs_smarts": lab.mcs_smarts,
                "mcs_size": lab.mcs_size,
                "top_smiles": " | ".join(lab.top_smiles[:5]),
            }
        )
    return pd.DataFrame(rows)


def save_labels(labels: list[FeatureLabel], out_dir: Path, tag: str = "labels") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = labels_to_dataframe(labels)
    df.to_csv(out_dir / f"{tag}.csv", index=False)
    # Full detail (incl. concept precisions) to JSON
    import json

    detail = [
        {
            "feature_id": lab.feature_id,
            "n_active": lab.n_active,
            "best_concept": lab.best_concept,
            "best_precision": lab.best_precision,
            "best_lift": lab.best_lift,
            "mcs_smarts": lab.mcs_smarts,
            "mcs_size": lab.mcs_size,
            "top_smiles": lab.top_smiles,
            "top_values": lab.top_values,
            "concept_precisions": lab.concept_precisions,
        }
        for lab in labels
    ]
    (out_dir / f"{tag}.json").write_text(json.dumps(detail, indent=2))


def top_tokens_per_feature(
    sae: SparseAutoencoder,
    tokens: torch.Tensor,
    token_meta: pd.DataFrame,
    smiles: list[str],
    feature_ids: list[int],
    k: int = 10,
) -> dict[int, pd.DataFrame]:
    """For each feature, return its top-K activating tokens with parent SMILES.

    Streams encoding through the SAE so the full [N_tokens, d_hidden] matrix
    never needs to be in memory.
    """
    if not feature_ids:
        return {}
    top_vals, top_idx = top_tokens_for_features(sae, tokens, feature_ids, k=k)
    mol_idx = token_meta["molecule_idx"].to_numpy()
    pos = token_meta["position"].to_numpy()
    tok_str = token_meta["token_str"].tolist()
    out: dict[int, pd.DataFrame] = {}
    for fi, fid in enumerate(feature_ids):
        rows = []
        for v, i in zip(top_vals[fi].tolist(), top_idx[fi].tolist()):
            mi = int(mol_idx[i])
            rows.append({
                "activation": float(v),
                "molecule_idx": mi,
                "smiles": smiles[mi],
                "token_position": int(pos[i]),
                "token_str": tok_str[i],
            })
        out[fid] = pd.DataFrame(rows)
    return out


def save_token_level(token_level: dict[int, pd.DataFrame], out_dir: Path, tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for fid, df in token_level.items():
        for _, r in df.iterrows():
            rows.append({"feature_id": fid, **r.to_dict()})
    pd.DataFrame(rows).to_csv(out_dir / f"{tag}_top_tokens.csv", index=False)


def encode_bundle_streaming(sae: SparseAutoencoder, bundle: ActivationBundle) -> torch.Tensor:
    """Memory-light: compute molecule-level max-over-tokens without materializing z_tokens."""
    mol_idx = bundle.token_meta["molecule_idx"].to_numpy()
    return molecule_level_max(sae, bundle.tokens, mol_idx, n_molecules=len(bundle.smiles))
