"""Build a training shard from leaves the search actually evaluated.

`teacher leaves` writes one JSON line per position the harness search asked about, with the
net's value and Stockfish's d9 value in pawns from the side to move. Those positions are the
deployment distribution: findings 68 and 71 measured the net's error on them, and it is much
larger than on the human positions it trains on.

The d9 value here is already the training target under the usual transform (clamped to ±2500
centipawns), so no relabeling is needed. Unlike chess_nn.label this keeps positions whose best
move is a capture: the search evaluates them, so the evaluator has to score them.

Usage:
  uv run python -m chess_nn.leaf_shard runs/teacher/leaves_train.jsonl --out-dir data/shards_leaf \\
      --prefix shard_leaf_a
"""

import argparse
import json
from pathlib import Path

import chess

from chess_nn.encoding import feature_indices
from chess_nn.shards import Row, write_shard

SCORE_LIMIT_CP = 2500


def rows_from(path: Path) -> list[Row]:
    rows: list[Row] = []
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        fen = record["fen"]
        key = " ".join(fen.split()[:4])
        if key in seen:
            continue
        seen.add(key)
        score = int(round(float(record["teacher"]) * 100))
        rows.append((fen, max(-SCORE_LIMIT_CP, min(SCORE_LIMIT_CP, score)), 0, feature_indices(chess.Board(fen))))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("leaves", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="shard_leaf")
    parser.add_argument("--shard-size", type=int, default=600_000)
    args = parser.parse_args()

    rows: list[Row] = []
    for path in args.leaves:
        rows += rows_from(path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for index, start in enumerate(range(0, len(rows), args.shard_size)):
        chunk = rows[start : start + args.shard_size]
        path = args.out_dir / f"{args.prefix}_{index:04d}.npz"
        write_shard(path, chunk)
        print(f"{path}: {len(chunk):,} positions")


if __name__ == "__main__":
    main()
