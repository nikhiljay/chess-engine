"""Static move-ranking accuracy of trained runs, split by pair kind.

A pair counts as correct when the net scores the better move's child lower,
i.e. on static evaluation alone it would prefer that move. Kinds come from
chess_nn.rank_pairs: repetition-rule pairs, and score-gap pairs split into
tactical (gap >= 300 cp) and moderate (100-299 cp).

Usage:
  uv run python -m chess_nn.pair_report data/rank_pairs_a_val.npz --runs runs/rank-a-h512-w0 runs/threat-a-h512-t64-w0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from chess_nn.encoding import dense_from_sparse
from chess_nn.model import EvalNet, model_from_config

TACTICAL_CP = 300


def load_run_model(run_dir: Path) -> EvalNet:
    config = json.loads((run_dir / "config.json").read_text())
    model = model_from_config(config)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    return model.eval()


@torch.no_grad()
def prefers_better(model: EvalNet, good: np.ndarray, bad: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    correct = []
    for start in range(0, len(good), batch_size):
        g = model(dense_from_sparse(torch.from_numpy(good[start : start + batch_size]))).squeeze(1)
        b = model(dense_from_sparse(torch.from_numpy(bad[start : start + batch_size]))).squeeze(1)
        correct.append((g < b).numpy())
    return np.concatenate(correct) if correct else np.zeros(0, dtype=bool)


def kind_masks(pairs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    repetition = pairs["repetition_rule"]
    mate = pairs["mate_rule"] if "mate_rule" in pairs else np.zeros(len(repetition), dtype=bool)
    score = ~repetition & ~mate
    gap = pairs["gap_cp"].astype(int)
    return {
        "mate-in-one rule": mate,
        "tactical score gap (>= 300 cp)": score & (gap >= TACTICAL_CP),
        "moderate score gap (100-299 cp)": score & (gap < TACTICAL_CP),
        "repetition rule": repetition,
        "all": np.ones(len(repetition), dtype=bool),
    }


def accuracy_by_kind(model: EvalNet, pairs: dict[str, np.ndarray]) -> dict[str, dict]:
    correct = prefers_better(model, pairs["good"], pairs["bad"])
    return {
        kind: {"pairs": int(mask.sum()), "accuracy": round(float(correct[mask].mean()), 3) if mask.any() else None}
        for kind, mask in kind_masks(pairs).items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pairs", type=Path, help="a rank_pairs .npz (usually the validation split)")
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    with np.load(args.pairs) as data:
        pairs = {key: data[key] for key in data.files}
    for run in args.runs:
        report = accuracy_by_kind(load_run_model(run), pairs)
        (run / f"pair_report_{args.pairs.stem}.json").write_text(json.dumps(report, indent=2))
        cells = "  ".join(f"{kind}: {r['accuracy']} (n={r['pairs']})" for kind, r in report.items())
        print(f"{run.name:34s} {cells}")


if __name__ == "__main__":
    main()
