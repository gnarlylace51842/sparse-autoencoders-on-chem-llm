"""Extract ChemBERTa hidden-state activations for a list of SMILES."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .config import ACTIVATIONS_DIR, CFG, DEVICE, DTYPE


@dataclass
class ActivationBundle:
    """Tensors and metadata produced by one extraction pass."""
    tokens: torch.Tensor          # [N_tokens, d_model], non-special tokens only
    token_meta: pd.DataFrame      # molecule_idx, position, token_id, token_str
    pooled: torch.Tensor          # [N_mol, d_model], mean of non-special tokens
    cls: torch.Tensor             # [N_mol, d_model]
    smiles: list[str]
    layer: int

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.tokens, out_dir / "tokens.pt")
        torch.save(self.pooled, out_dir / "pooled.pt")
        torch.save(self.cls, out_dir / "cls.pt")
        self.token_meta.to_parquet(out_dir / "token_meta.parquet")
        pd.DataFrame({"smiles": self.smiles}).to_parquet(out_dir / "molecules.parquet")
        (out_dir / "layer.txt").write_text(str(self.layer))

    @classmethod
    def load(cls, out_dir: Path) -> "ActivationBundle":
        return cls(
            tokens=torch.load(out_dir / "tokens.pt", map_location="cpu"),
            token_meta=pd.read_parquet(out_dir / "token_meta.parquet"),
            pooled=torch.load(out_dir / "pooled.pt", map_location="cpu"),
            cls=torch.load(out_dir / "cls.pt", map_location="cpu"),
            smiles=pd.read_parquet(out_dir / "molecules.parquet")["smiles"].tolist(),
            layer=int((out_dir / "layer.txt").read_text().strip()),
        )


def _load_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(CFG.model.name)
    model = AutoModel.from_pretrained(CFG.model.name)
    model.eval()
    model.to(DEVICE)
    return model, tok


@torch.no_grad()
def extract_activations(
    smiles: list[str],
    layer: int | None = None,
    batch_size: int | None = None,
    max_length: int | None = None,
) -> ActivationBundle:
    """Run ChemBERTa, return non-special token activations + pooled summaries."""
    layer = CFG.activation.layer if layer is None else layer
    batch_size = batch_size or CFG.activation.batch_size
    max_length = max_length or CFG.model.max_length

    model, tok = _load_model_and_tokenizer()
    d_model = model.config.hidden_size

    tokens_chunks: list[torch.Tensor] = []
    pooled = torch.empty(len(smiles), d_model, dtype=DTYPE)
    cls = torch.empty(len(smiles), d_model, dtype=DTYPE)
    meta_rows: list[tuple[int, int, int, str]] = []

    # token_meta records position_within_molecule for non-special tokens only,
    # so SAE feature analysis can map back to a specific token in a SMILES.
    special_ids = set(tok.all_special_ids)

    for start in tqdm(range(0, len(smiles), batch_size), desc=f"layer {layer}"):
        batch = smiles[start : start + batch_size]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(DEVICE)

        out = model(**enc, output_hidden_states=True)
        # hidden_states is a tuple of length n_layers+1: embeddings + each layer's output
        hs = out.hidden_states[layer].to(DTYPE)  # [B, L, d]
        attn = enc["attention_mask"].bool()
        input_ids = enc["input_ids"]

        cls[start : start + len(batch)] = hs[:, 0, :].cpu()

        for i in range(len(batch)):
            mol_idx = start + i
            row_mask = attn[i]
            row_ids = input_ids[i]
            row_hs = hs[i]  # [L, d]

            # mark non-special, non-pad positions
            keep = row_mask.clone()
            for sid in special_ids:
                keep &= row_ids != sid

            kept_hs = row_hs[keep].cpu()                     # [n_kept, d]
            kept_ids = row_ids[keep].cpu().tolist()
            positions = torch.nonzero(keep, as_tuple=True)[0].cpu().tolist()

            tokens_chunks.append(kept_hs)
            pooled[mol_idx] = kept_hs.mean(dim=0) if kept_hs.numel() > 0 else 0.0

            for pos, tid in zip(positions, kept_ids):
                meta_rows.append((mol_idx, pos, tid, tok.convert_ids_to_tokens(tid)))

    tokens = torch.cat(tokens_chunks, dim=0)
    meta = pd.DataFrame(
        meta_rows,
        columns=["molecule_idx", "position", "token_id", "token_str"],
    )
    return ActivationBundle(
        tokens=tokens,
        token_meta=meta,
        pooled=pooled,
        cls=cls,
        smiles=smiles,
        layer=layer,
    )


def activations_dir_for_layer(layer: int) -> Path:
    return ACTIVATIONS_DIR / f"layer_{layer:02d}"
