"""Leaf error by regime: how far a model sits from the d9 targets where the search looks.

Scores the positions a search actually evaluated (from `teacher leaves`, which writes
{"fen", "net", "teacher"} per line) with any trained model, and reports bias and MAE
split by the teacher's value. Finding 68 measured the default this way: accurate when
level or losing, 4-6 pawns off when winning, which is over half the leaves it evaluates.

Usage:
  uv run python -m chess_nn.leaf_error runs/teacher/leaves_d9_dev5.jsonl runs/jj6-h1024-s1000-cp20 runs/fid-a-h1024-cp10
"""

import argparse
import json
from pathlib import Path

import chess
import numpy as np

from chess_nn.export import OnnxEvaluator

BUCKETS = [(-99.0, -10.0), (-10.0, -3.0), (-3.0, -1.0), (-1.0, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 99.0)]


def label(low: float, high: float) -> str:
    return f"{low:+.0f}..{high:+.0f}".replace("-99", "  -inf").replace("+99", "inf")


def report(leaves: Path, runs: list[Path]) -> None:
    rows = [json.loads(line) for line in leaves.read_text().splitlines() if line.strip()]
    boards = [chess.Board(r["fen"]) for r in rows]
    teacher = np.array([r["teacher"] for r in rows], dtype=np.float64)
    print(f"{len(rows):,} leaves from {leaves}")

    for run in runs:
        values = np.asarray(OnnxEvaluator(run / "model.onnx").evaluate(boards), dtype=np.float64)
        error = values - teacher
        overall = f"bias {error.mean():+.2f}  MAE {np.abs(error).mean():.2f}  RMSE {np.sqrt((error**2).mean()):.2f}"
        print(f"\n{run.name}: {overall}")
        for low, high in BUCKETS:
            inside = (teacher >= low) & (teacher < high)
            if inside.sum() < 20:
                continue
            group = error[inside]
            print(f"  {label(low, high):>12s}  n={inside.sum():6d}  bias {group.mean():+6.2f}  MAE {np.abs(group).mean():5.2f}")
        # Level positions decide games, and over-dispersion there is what drives blunders:
        # predicted spread against the truth's, plus how often the model is badly wrong.
        level = (teacher >= -3) & (teacher <= 3)
        if level.sum() > 20:
            predicted, truth = values[level], teacher[level]
            correlation = float(np.corrcoef(predicted, truth)[0, 1])
            big = float((np.abs(predicted - truth) > 3).mean())
            print(f"  level (-3..+3): sd(pred) {predicted.std():.2f} vs sd(true) {truth.std():.2f}, "
                  f"r {correlation:.3f}, |err|>3 {100 * big:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("leaves", type=Path, help="jsonl from `teacher leaves`")
    parser.add_argument("runs", type=Path, nargs="+", help="run directories holding model.onnx")
    args = parser.parse_args()
    report(args.leaves, args.runs)


if __name__ == "__main__":
    main()
