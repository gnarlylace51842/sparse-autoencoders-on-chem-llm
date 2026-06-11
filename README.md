# sparse-autoencoders-on-chem-llm

Code for a paper applying sparse autoencoders to ChemBERTa
(`seyonec/ChemBERTa-zinc-base-v1`). Trains an SAE on per-token
activations, checks which features line up with named chemical
concepts (functional groups, ring systems, halogens, chirality).

Apple Silicon MPS, float32. CUDA not tested.

## Install

```
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Run

Numbered scripts, run in order. Full layer-6 run is ~50 min on M4 Max.

```
scripts/01_extract_activations.py    ChemBERTa -> activations
scripts/02_train_sae.py              train SAE
scripts/03_analyze_features.py       top-K mols + SMARTS labeling
scripts/04_specificity_tests.py      paired-molecule sanity checks
scripts/05_layer_comparison.py       layers 2, 4, 6
scripts/06_shuffled_control.py       shuffled-activation control
scripts/07_neuron_baseline.py        raw-neuron baseline
scripts/08_make_paper_artifacts.py   figures, tables, RESULTS.md
```

`scripts/00_smoke_test.py` runs a tiny end-to-end check (~2 min) if
you want to sanity-check the install before the real run.

All flags default to `chemsae/config.py`; override on the CLI
(`--layer`, `--n`, `--steps`, `--l1`, `--seed`, `--tag`).

## Layout

```
chemsae/      package
scripts/      numbered run scripts
paper/        paper.md, paper.docx
results/      figures/, tables/, per-layer label CSV/JSON, RESULTS.md
```

`data/`, `activations/`, `checkpoints/`, and the big `results/z_*.pt`
encoded tensors are gitignored. They regenerate from scripts 01-02
(plus 03 if you want the encoded tensors back).

## What the SAE does

```
z     = ReLU(W_enc (x - b_dec) + b_enc)
x_hat = W_dec z + b_dec
loss  = MSE(x_hat, x) + λ ||z||_1
```

4x overcomplete (3072 features for 768-dim input), decoder columns
renormalized to unit norm after each step. For each feature, take its
top-20 activating molecules and match against 44 curated SMARTS
patterns. Feature gets a concept label if precision >= 0.7 and
lift >= 1.3 (or precision >= 0.95). The same labeling pipeline runs on
raw ChemBERTa neurons in script 07 for comparison.

## Reproducing the paper numbers

```
for L in 2 4 6; do
  python scripts/01_extract_activations.py --layer $L --n 50000
  python scripts/02_train_sae.py --layer $L
  python scripts/03_analyze_features.py --layer $L
  python scripts/07_neuron_baseline.py --layer $L
done
python scripts/04_specificity_tests.py --layer 6
python scripts/06_shuffled_control.py --layer 6
python scripts/02_train_sae.py --layer 6 --tag sae_seed1 --seed 1
python scripts/03_analyze_features.py --layer 6 --tag sae_seed1
python scripts/05_layer_comparison.py --layers 2 4 6
python scripts/08_make_paper_artifacts.py --layer 6
```

Default seed is 0 throughout. The seed-stability section in the paper
comes from the `--seed 1` run above.
