"""Step 4: specificity tests on curated paired molecules.

For each (concept, positive, negative) triple, take the feature whose
labeling matched `concept` from step 3 (if any) and check whether its
activation on the positive molecule exceeds its activation on the negative.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from chemsae.config import CFG, CHECKPOINTS_DIR, RESULTS_DIR, ensure_dirs
from chemsae.controls import run_specificity_tests
from chemsae.training import load_sae


def _load_concept_map(layer: int, tag: str) -> dict[str, int]:
    """Build concept -> feature_id, choosing the entry with best (precision * lift)."""
    path = RESULTS_DIR / f"layer_{layer:02d}" / f"{tag}.json"
    if not path.exists():
        return {}
    detail = json.loads(path.read_text())
    best: dict[str, dict] = {}
    for entry in detail:
        c = entry.get("best_concept")
        if not c:
            continue
        score = entry["best_precision"] * entry["best_lift"]
        if c not in best or best[c]["best_precision"] * best[c]["best_lift"] < score:
            best[c] = entry
    return {c: e["feature_id"] for c, e in best.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=CFG.activation.layer)
    parser.add_argument("--tag", type=str, default="sae")
    args = parser.parse_args()

    ensure_dirs()
    sae = load_sae(CHECKPOINTS_DIR / f"layer_{args.layer:02d}" / f"{args.tag}_final.pt")
    concept_map = _load_concept_map(args.layer, args.tag)
    print(f"loaded {len(concept_map)} labeled concept->feature mappings")

    df = run_specificity_tests(sae, layer=args.layer, feature_concept_map=concept_map)
    out_path = RESULTS_DIR / f"layer_{args.layer:02d}" / f"{args.tag}_specificity.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
