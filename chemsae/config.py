"""Central configuration for the SAE-on-ChemBERTa pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ACTIVATIONS_DIR = PROJECT_ROOT / "activations"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = select_device()
DTYPE = torch.float32  # MPS is most stable in float32


@dataclass(frozen=True)
class ModelConfig:
    name: str = "seyonec/ChemBERTa-zinc-base-v1"
    d_model: int = 768
    n_layers: int = 6              # ChemBERTa-zinc-base-v1 is a 6-layer RoBERTa
    max_length: int = 128


@dataclass(frozen=True)
class DataConfig:
    source: str = "sagawa/ZINC-canonicalized"  # HF dataset
    split: str = "train"
    n_molecules: int = 20_000
    smiles_column: str = "smiles"
    seed: int = 0


@dataclass(frozen=True)
class ActivationConfig:
    layer: int = 6                  # 0..11, 6 is the canonical middle layer
    pooling: str = "tokens"         # "tokens" | "mean" | "cls"
    batch_size: int = 64
    drop_special_tokens: bool = True  # drop [CLS], [SEP], [PAD] for pooling="tokens"


@dataclass(frozen=True)
class SAEConfig:
    expansion: int = 4              # hidden = expansion * d_model
    l1_coeff: float = 1e-3
    lr: float = 1e-4
    batch_size: int = 1024
    n_steps: int = 30_000
    warmup_steps: int = 1_000
    log_every: int = 200
    save_every: int = 5_000
    seed: int = 0
    tied_init: bool = True          # initialize decoder = encoder.T, untie afterwards


@dataclass(frozen=True)
class AnalysisConfig:
    top_k: int = 20                 # top-K molecules per feature
    n_features_to_label: int = 150  # cap on features inspected
    min_activation: float = 1e-3    # ignore features that never activate
    mcs_timeout: int = 3            # RDKit MCS search timeout (seconds)


@dataclass(frozen=True)
class PipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    activation: ActivationConfig = field(default_factory=ActivationConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


CFG = PipelineConfig()


def ensure_dirs() -> None:
    for d in (DATA_DIR, ACTIVATIONS_DIR, CHECKPOINTS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
