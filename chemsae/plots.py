"""Matplotlib figures for the paper.

Each function takes a path or dataframe, returns a matplotlib Figure, and is
called by `scripts/08_make_figures.py`. Default style is intentionally plain:
no per-figure styling, no rcParams munging — figures should look the same
across environments.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_curves(history_path: Path) -> plt.Figure:
    """SAE training: MSE, L0, dead-feature count over steps."""
    rows = json.loads(history_path.read_text())
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    axes[0].plot(df["step"], df["mse"])
    axes[0].set(xlabel="step", ylabel="reconstruction MSE", title="reconstruction loss")
    axes[0].set_yscale("log")
    axes[1].plot(df["step"], df["l0"])
    axes[1].set(xlabel="step", ylabel="active features per sample (L0)", title="sparsity")
    axes[2].plot(df["step"], df["n_dead_2k"])
    axes[2].set(xlabel="step", ylabel="dead features (no fire in 2k steps)", title="dead features")
    return fig


def plot_lift_histogram(labels_path: Path, shuffled_labels_path: Path | None = None) -> plt.Figure:
    """Distribution of best_lift per feature. Optionally overlay the shuffled control."""
    df = pd.read_csv(labels_path)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    lifts = df["best_lift"].fillna(0.0)
    ax.hist(lifts, bins=30, alpha=0.7, label="real SAE", color="C0")
    if shuffled_labels_path and shuffled_labels_path.exists():
        sdf = pd.read_csv(shuffled_labels_path)
        ax.hist(sdf["best_lift"].fillna(0.0), bins=30, alpha=0.6, label="shuffled control", color="C3")
    ax.axvline(2.0, color="k", linestyle="--", linewidth=1, label="lift = 2 threshold")
    ax.set(xlabel="best concept lift", ylabel="# features", title="feature interpretability lift")
    ax.legend()
    return fig


def plot_per_concept_best(labels_path: Path, top_n: int = 25) -> plt.Figure:
    """For each detected concept, plot the best feature's lift."""
    df = pd.read_csv(labels_path)
    labeled = df.dropna(subset=["best_concept"])
    if labeled.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no labeled features", ha="center", va="center")
        return fig
    best = labeled.sort_values("best_lift", ascending=False).groupby("best_concept", as_index=False).first()
    best = best.sort_values("best_lift", ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.25 * len(best))), constrained_layout=True)
    ax.barh(best["best_concept"], best["best_lift"], color="C0")
    ax.set(xlabel="best feature lift", title="best SAE feature per chemical concept")
    return fig


def plot_layer_comparison(summary_path: Path, results_dir: Path | None = None) -> plt.Figure:
    """Per-layer selectivity: count of features with lift>=10, max lift, unique concepts.

    Reads the summary CSV for layer membership and (if available) per-layer
    per-feature CSVs for richer counts. Falls back to summary-only if those
    aren't present.
    """
    summary = pd.read_csv(summary_path).sort_values("layer")
    rows = []
    base = results_dir or summary_path.parent
    for _, r in summary.iterrows():
        layer = int(r["layer"])
        sae_csv = base / f"layer_{layer:02d}" / "sae.csv"
        neu_csv = base / f"layer_{layer:02d}" / "neurons.csv"
        sae = pd.read_csv(sae_csv).dropna(subset=["best_concept"]) if sae_csv.exists() else None
        neu = pd.read_csv(neu_csv).dropna(subset=["best_concept"]) if neu_csv.exists() else None
        rows.append({
            "layer": layer,
            "sae_lift10": int((sae["best_lift"] >= 10).sum()) if sae is not None else int(r.get("n_labeled", 0)),
            "neu_lift10": int((neu["best_lift"] >= 10).sum()) if neu is not None else 0,
            "sae_max_lift": float(sae["best_lift"].max()) if sae is not None else float(r.get("max_lift", 0)),
            "neu_max_lift": float(neu["best_lift"].max()) if neu is not None else 0.0,
            "sae_concepts": int(sae["best_concept"].nunique()) if sae is not None else int(r.get("unique_concepts", 0)),
            "neu_concepts": int(neu["best_concept"].nunique()) if neu is not None else 0,
        })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    x = np.arange(len(df))
    w = 0.35
    axes[0].bar(x - w/2, df["sae_lift10"], width=w, label="SAE", color="C0")
    axes[0].bar(x + w/2, df["neu_lift10"], width=w, label="raw neurons", color="C7")
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"layer {l}" for l in df["layer"]])
    axes[0].set(ylabel="# features with lift ≥ 10", title="highly-selective features per layer")
    axes[0].legend()

    axes[1].plot(df["layer"], df["sae_max_lift"], marker="o", label="SAE", color="C0")
    axes[1].plot(df["layer"], df["neu_max_lift"], marker="s", label="raw neurons", color="C7")
    axes[1].set(xlabel="layer", ylabel="max lift", title="best feature lift across depth")
    axes[1].set_xticks(df["layer"])
    axes[1].legend()

    axes[2].bar(x - w/2, df["sae_concepts"], width=w, label="SAE", color="C0")
    axes[2].bar(x + w/2, df["neu_concepts"], width=w, label="raw neurons", color="C7")
    axes[2].set_xticks(x); axes[2].set_xticklabels([f"layer {l}" for l in df["layer"]])
    axes[2].set(ylabel="# unique concepts", title="concept diversity across depth")
    axes[2].legend()
    return fig


def plot_specificity(specificity_path: Path) -> plt.Figure:
    """Per (concept, pos, neg) triple, plot pos vs neg activations side-by-side."""
    df = pd.read_csv(specificity_path)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.25 * len(df))), constrained_layout=True)
    y = np.arange(len(df))
    ax.barh(y - 0.2, df["pos_activation"], height=0.4, label="positive", color="C0")
    ax.barh(y + 0.2, df["neg_activation"], height=0.4, label="negative", color="C3")
    labels = [f"{r['concept']}  ({r['pos_name']} vs {r['neg_name']})" for _, r in df.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set(xlabel="SAE feature activation", title="specificity tests: positive vs negative molecule")
    ax.legend()
    return fig


def plot_concept_by_layer_heatmap(matrix_csv: Path) -> plt.Figure:
    """Heatmap: concept (rows) × layer (cols), values = best-feature lift.

    Cell color = lift over baseline (log scale). Cells with value 0 mean the
    concept was not detected by any feature in that layer's top-200.
    """
    df = pd.read_csv(matrix_csv)
    layers = [c for c in df.columns if c.startswith("layer_")]
    M = df[layers].to_numpy()
    concepts = df["concept"].tolist()

    fig, ax = plt.subplots(figsize=(5.5, max(6, 0.27 * len(concepts))), constrained_layout=True)
    im = ax.imshow(
        np.log10(np.maximum(M, 1.0)),
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([l.replace("layer_", "layer ") for l in layers])
    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts, fontsize=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("log10(best feature lift)")

    # Annotate non-zero cells with the lift value
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v >= 0.5:
                color = "white" if v < 5 else "black"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", color=color, fontsize=6.5)

    ax.set_title("best SAE feature lift per concept, per layer")
    return fig


def plot_neurons_vs_sae(neurons_path: Path, sae_path: Path) -> plt.Figure:
    """Compare selectivity (high-lift features) between raw neurons and SAE.

    Median lift differences between neurons and SAE are tiny (1.36 vs 1.57);
    the SAE advantage shows up in the *tail* — the count of features that
    pass a high-lift bar (≥ 5, ≥ 10), and the diversity of concepts captured.
    """
    def _read(p: Path) -> pd.DataFrame | None:
        if not p.exists():
            return None
        return pd.read_csv(p).dropna(subset=["best_concept"])

    neu = _read(neurons_path)
    sae = _read(sae_path)
    if neu is None or sae is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "missing input", ha="center")
        return fig

    bars = {
        "lift ≥ 5": [int((neu["best_lift"] >= 5).sum()), int((sae["best_lift"] >= 5).sum())],
        "lift ≥ 10": [int((neu["best_lift"] >= 10).sum()), int((sae["best_lift"] >= 10).sum())],
        "unique concepts": [int(neu["best_concept"].nunique()), int(sae["best_concept"].nunique())],
    }

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for ax, (title, vals) in zip(axes, bars.items()):
        ax.bar(["raw neurons", "SAE features"], vals, color=["C7", "C0"])
        ax.set(ylabel="count", title=title)
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=11)
    return fig


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_feature_grid(
    labels_json_path: Path, output_path: Path, n_features: int = 8, mols_per_feature: int = 4
) -> None:
    """Render a grid of top-K activating molecules for the top N labeled features.

    For each selected feature, highlights the substructure that matches the
    labeled concept's SMARTS. Output is a single PNG that reads top-to-bottom
    as features ranked by best_lift, left-to-right as top-activating molecules.
    """
    import json
    from io import BytesIO

    from PIL import Image
    from rdkit import Chem
    from rdkit.Chem import Draw

    from chemsae.concepts import SMARTS

    data = json.loads(labels_json_path.read_text())
    feats = sorted(
        [d for d in data if d.get("best_concept")],
        key=lambda d: -d["best_lift"],
    )[:n_features]

    rows = []
    smarts_compiled = {n: Chem.MolFromSmarts(s) for n, s in SMARTS.items()}
    for d in feats:
        concept = d["best_concept"]
        pat = smarts_compiled.get(concept)
        smiles_list = d["top_smiles"][:mols_per_feature]
        mols, highlights = [], []
        for s in smiles_list:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            mols.append(m)
            if pat is not None:
                match = m.GetSubstructMatch(pat)
                highlights.append(list(match))
            else:
                highlights.append([])
        if not mols:
            continue
        legends = [f"act={v:.2f}" for v in d["top_values"][: len(mols)]]
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=mols_per_feature,
            subImgSize=(280, 220),
            legends=legends,
            highlightAtomLists=highlights,
            useSVG=False,
        )
        # convert PIL image -> numpy via bytes
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        rows.append((d, buf))

    if not rows:
        return

    # Compose vertically: each row = feature title + the grid image
    images = [Image.open(buf) for _, buf in rows]
    title_h = 36
    pad = 4
    total_w = max(im.width for im in images)
    total_h = sum(im.height + title_h + pad for im in images)
    canvas = Image.new("RGB", (total_w, total_h), "white")

    from PIL import ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except Exception:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)
    y = 0
    for (d, _buf), im in zip(rows, images):
        title = f"feature {d['feature_id']}: {d['best_concept']}  (precision {d['best_precision']:.2f}, lift {d['best_lift']:.2f})"
        draw.text((6, y + 6), title, fill="black", font=font)
        canvas.paste(im, (0, y + title_h))
        y += title_h + im.height + pad

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
