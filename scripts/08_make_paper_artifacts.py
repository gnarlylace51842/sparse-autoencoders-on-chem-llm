"""Step 8: aggregate results into paper-ready figures and tables.

Reads everything that earlier scripts wrote under `results/` and
`checkpoints/`, produces:

  results/figures/                — PNG figures
  results/tables/                  — CSV summary tables
  results/RESULTS.md               — human-readable findings summary

Designed to be re-run any time after upstream scripts have produced new files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from chemsae import plots
from chemsae.config import CFG, CHECKPOINTS_DIR, PROJECT_ROOT, RESULTS_DIR, ensure_dirs


FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"


def _exists(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def _read_labels(layer: int, tag: str) -> pd.DataFrame | None:
    p = RESULTS_DIR / f"layer_{layer:02d}" / f"{tag}.csv"
    return pd.read_csv(p) if _exists(p) else None


def per_concept_table(df: pd.DataFrame) -> pd.DataFrame:
    """Best feature per labeled concept (ranked by precision × log(lift+1))."""
    labeled = df.dropna(subset=["best_concept"]).copy()
    if labeled.empty:
        return labeled
    import numpy as np
    labeled["score"] = labeled["best_precision"] * np.log1p(labeled["best_lift"])
    best = (
        labeled.sort_values("score", ascending=False)
        .groupby("best_concept", as_index=False)
        .agg(
            n_features=("feature_id", "count"),
            best_feature_id=("feature_id", "first"),
            best_precision=("best_precision", "first"),
            best_lift=("best_lift", "first"),
            top_smiles=("top_smiles", "first"),
        )
        .sort_values("best_lift", ascending=False)
        .reset_index(drop=True)
    )
    return best


def headline_summary(layer: int, sae_tag: str = "sae", shuffled_tag: str = "sae_shuffled",
                     neurons_tag: str = "neurons") -> dict:
    sae = _read_labels(layer, sae_tag)
    shuffled = _read_labels(layer, shuffled_tag)
    neurons = _read_labels(layer, neurons_tag)

    def _summary(df: pd.DataFrame | None) -> dict:
        if df is None:
            return {"available": False}
        labeled = df.dropna(subset=["best_concept"])
        return {
            "available": True,
            "n_features": int(len(df)),
            "n_labeled": int(len(labeled)),
            "frac_labeled": float(len(labeled) / max(len(df), 1)),
            "unique_concepts": int(labeled["best_concept"].nunique()) if len(labeled) else 0,
            "median_precision": float(labeled["best_precision"].median()) if len(labeled) else float("nan"),
            "median_lift": float(labeled["best_lift"].median()) if len(labeled) else float("nan"),
            "max_lift": float(labeled["best_lift"].max()) if len(labeled) else float("nan"),
        }

    return {
        "layer": layer,
        "sae": _summary(sae),
        "shuffled": _summary(shuffled),
        "neurons": _summary(neurons),
    }


def write_tables(layer: int) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    sae = _read_labels(layer, "sae")
    if sae is not None:
        sae.sort_values("best_lift", ascending=False, na_position="last").to_csv(
            TABLE_DIR / f"layer_{layer:02d}_features.csv", index=False
        )
        per_concept_table(sae).to_csv(TABLE_DIR / f"layer_{layer:02d}_per_concept.csv", index=False)

    # Per-concept × per-layer matrix
    cross_layer_rows: dict[str, dict[int, float]] = {}
    for l in (2, 4, 6):
        df = _read_labels(l, "sae")
        if df is None:
            continue
        labeled = df.dropna(subset=["best_concept"])
        for _, r in labeled.iterrows():
            concept = r["best_concept"]
            cross_layer_rows.setdefault(concept, {})
            if l not in cross_layer_rows[concept] or cross_layer_rows[concept][l] < r["best_lift"]:
                cross_layer_rows[concept][l] = float(r["best_lift"])
    if cross_layer_rows:
        rows = []
        for concept, layer_lifts in cross_layer_rows.items():
            rows.append({
                "concept": concept,
                "layer_2": layer_lifts.get(2, 0.0),
                "layer_4": layer_lifts.get(4, 0.0),
                "layer_6": layer_lifts.get(6, 0.0),
            })
        import pandas as _pd
        mdf = _pd.DataFrame(rows)
        mdf["best_layer"] = mdf[["layer_2", "layer_4", "layer_6"]].idxmax(axis=1).str.replace("layer_", "").astype(int)
        mdf["best_lift"] = mdf[["layer_2", "layer_4", "layer_6"]].max(axis=1)
        mdf.sort_values("best_lift", ascending=False).to_csv(TABLE_DIR / "per_concept_per_layer.csv", index=False)

    summary_path = RESULTS_DIR / "layer_comparison_summary.csv"
    if _exists(summary_path):
        pd.read_csv(summary_path).to_csv(TABLE_DIR / "layer_comparison.csv", index=False)

    spec_path = RESULTS_DIR / f"layer_{layer:02d}" / "sae_specificity.csv"
    if _exists(spec_path):
        pd.read_csv(spec_path).to_csv(TABLE_DIR / f"layer_{layer:02d}_specificity.csv", index=False)


def write_figures(layer: int) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    history = CHECKPOINTS_DIR / f"layer_{layer:02d}" / "sae_history.json"
    if _exists(history):
        plots.save_fig(plots.plot_training_curves(history), FIG_DIR / f"fig_training_layer{layer:02d}.png")

    sae_labels = RESULTS_DIR / f"layer_{layer:02d}" / "sae.csv"
    shuffled_labels = RESULTS_DIR / f"layer_{layer:02d}" / "sae_shuffled.csv"
    if _exists(sae_labels):
        plots.save_fig(
            plots.plot_lift_histogram(sae_labels, shuffled_labels if _exists(shuffled_labels) else None),
            FIG_DIR / f"fig_lift_layer{layer:02d}.png",
        )
        plots.save_fig(plots.plot_per_concept_best(sae_labels), FIG_DIR / f"fig_per_concept_layer{layer:02d}.png")

    spec_path = RESULTS_DIR / f"layer_{layer:02d}" / "sae_specificity.csv"
    if _exists(spec_path):
        plots.save_fig(plots.plot_specificity(spec_path), FIG_DIR / f"fig_specificity_layer{layer:02d}.png")

    summary_path = RESULTS_DIR / "layer_comparison_summary.csv"
    if _exists(summary_path):
        plots.save_fig(plots.plot_layer_comparison(summary_path), FIG_DIR / "fig_layer_comparison.png")

    neurons_path = RESULTS_DIR / f"layer_{layer:02d}" / "neurons.csv"
    if _exists(neurons_path) and _exists(sae_labels):
        plots.save_fig(plots.plot_neurons_vs_sae(neurons_path, sae_labels), FIG_DIR / f"fig_neurons_vs_sae_layer{layer:02d}.png")

    sae_json = RESULTS_DIR / f"layer_{layer:02d}" / "sae.json"
    if _exists(sae_json):
        plots.render_feature_grid(
            sae_json,
            FIG_DIR / f"fig_feature_grid_layer{layer:02d}.png",
            n_features=8,
            mols_per_feature=4,
        )

    per_concept_layer_path = TABLE_DIR / "per_concept_per_layer.csv"
    if _exists(per_concept_layer_path):
        plots.save_fig(
            plots.plot_concept_by_layer_heatmap(per_concept_layer_path),
            FIG_DIR / "fig_concept_by_layer.png",
        )


def write_markdown_report(layer: int) -> None:
    head = headline_summary(layer)
    md = ["# Results summary", ""]

    # Read actual molecule count from the cached parquet rather than the
    # config default, so re-runs with different --n correctly report the
    # corpus size that produced these results.
    smi_cache = PROJECT_ROOT / "data" / "smiles.parquet"
    try:
        import pandas as pd
        n_actual = len(pd.read_parquet(smi_cache, columns=["smiles"]))
    except Exception:
        n_actual = CFG.data.n_molecules

    md.append(f"Layer: **{layer}**, model: `{CFG.model.name}`, "
              f"corpus: `{CFG.data.source}` × {n_actual}, "
              f"SAE expansion: **{CFG.sae.expansion}×**, L1: **{CFG.sae.l1_coeff}**.")
    md.append("")
    md.append("## Headline")
    md.append("")
    md.append("| Variant | features | labeled | frac labeled | unique concepts | median lift | max lift |")
    md.append("|---------|----------|---------|--------------|-----------------|-------------|----------|")
    for name, key in [("SAE", "sae"), ("shuffled control", "shuffled"), ("raw neurons", "neurons")]:
        s = head[key]
        if not s["available"]:
            md.append(f"| {name} | — | — | — | — | — | — |")
        else:
            md.append(
                f"| {name} | {s['n_features']} | {s['n_labeled']} | {s['frac_labeled']:.2f} | "
                f"{s['unique_concepts']} | {s['median_lift']:.2f} | {s['max_lift']:.2f} |"
            )
    md.append("")

    sae = _read_labels(layer, "sae")
    if sae is not None:
        per_concept = per_concept_table(sae)
        md.append("## Best feature per chemical concept")
        md.append("")
        md.append("| Concept | # features | best feature | precision | lift | top SMILES (truncated) |")
        md.append("|---------|------------|--------------|-----------|------|------------------------|")
        for _, r in per_concept.head(40).iterrows():
            top = (r["top_smiles"] or "")[:80]
            md.append(
                f"| {r['best_concept']} | {r['n_features']} | "
                f"{int(r['best_feature_id'])} | {r['best_precision']:.2f} | {r['best_lift']:.2f} | `{top}` |"
            )
        md.append("")

    summary_path = RESULTS_DIR / "layer_comparison_summary.csv"
    if _exists(summary_path):
        md.append("## Layer comparison")
        md.append("")
        md.append((pd.read_csv(summary_path)).to_markdown(index=False))
        md.append("")

    spec_path = RESULTS_DIR / f"layer_{layer:02d}" / "sae_specificity.csv"
    if _exists(spec_path):
        md.append("## Specificity tests")
        md.append("")
        md.append(pd.read_csv(spec_path).to_markdown(index=False))
        md.append("")

    md.append("## Figures")
    md.append("")
    for fname in sorted(FIG_DIR.glob("*.png")):
        md.append(f"- `{fname.relative_to(RESULTS_DIR)}`")
    md.append("")

    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(md))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=CFG.activation.layer)
    args = parser.parse_args()

    ensure_dirs()
    write_tables(args.layer)
    write_figures(args.layer)
    write_markdown_report(args.layer)
    print(f"wrote artifacts under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
