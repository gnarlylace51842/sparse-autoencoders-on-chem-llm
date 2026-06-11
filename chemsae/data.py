"""SMILES dataset loading and canonicalization."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

from .config import CFG, DATA_DIR

RDLogger.DisableLog("rdApp.*")  # silence RDKit SMILES parse warnings


def _canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def canonicalize_many(smiles: Iterable[str]) -> list[str]:
    out: list[str] = []
    for s in smiles:
        canon = _canonicalize(s)
        if canon is not None:
            out.append(canon)
    return out


def load_smiles(
    n: int | None = None,
    source: str | None = None,
    cache_name: str = "smiles.parquet",
    seed: int | None = None,
) -> list[str]:
    """Load `n` canonical SMILES. Caches to data/smiles.parquet."""
    n = n or CFG.data.n_molecules
    source = source or CFG.data.source
    seed = CFG.data.seed if seed is None else seed
    cache = DATA_DIR / cache_name

    if cache.exists():
        df = pd.read_parquet(cache)
        if len(df) >= n:
            return df["smiles"].head(n).tolist()

    smiles = _download_smiles(source, n, seed)
    smiles = canonicalize_many(smiles)
    smiles = list(dict.fromkeys(smiles))  # dedupe, preserve order
    smiles = smiles[:n]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"smiles": smiles}).to_parquet(cache)
    return smiles


def _download_smiles(source: str, n: int, seed: int) -> list[str]:
    """Stream from HuggingFace; oversample then filter for canonicalization losses."""
    from datasets import load_dataset

    over = int(n * 1.3) + 100  # oversample so we have headroom after dedupe/canon
    ds = load_dataset(source, split=CFG.data.split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    col = CFG.data.smiles_column
    smiles: list[str] = []
    pbar = tqdm(total=over, desc=f"streaming {source}")
    for row in ds:
        s = row.get(col) or row.get("SMILES") or row.get("canonical_smiles")
        if not s:
            continue
        smiles.append(s)
        pbar.update(1)
        if len(smiles) >= over:
            break
    pbar.close()
    return smiles


def load_from_csv(path: str | Path, column: str = "smiles", n: int | None = None) -> list[str]:
    """Escape hatch: load SMILES from a local CSV when HuggingFace is unavailable."""
    df = pd.read_csv(path)
    smiles = df[column].dropna().astype(str).tolist()
    smiles = canonicalize_many(smiles)
    smiles = list(dict.fromkeys(smiles))
    if n is not None:
        smiles = smiles[:n]
    return smiles
