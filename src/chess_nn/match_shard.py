"""Rebuild a shard over exactly the positions another shard kept, with the original labels.

Comparing label depths needs both arms trained on the same positions: relabeling at a greater
depth drops whatever no longer has a quiet best move, so the deeper arm is a subset. This
writes the shallower arm restricted to that subset, leaving label depth as the only difference.

Usage:
  uv run python -m chess_nn.match_shard --like data/shards_d12 --from data/shards \\
      --out-dir data/shards_d9_matched --prefix shard_d9m
"""

import argparse
from pathlib import Path

import numpy as np

from chess_nn.shards import load_fens, load_shards, write_shard


def key(fen: bytes | str) -> str:
    text = fen.decode() if isinstance(fen, (bytes, np.bytes_)) else str(fen)
    return " ".join(text.split()[:4])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--like", type=Path, required=True, help="directory whose positions define the subset")
    parser.add_argument("--from", dest="source", type=Path, nargs="+", required=True, help="shard files to draw labels from")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="shard_matched")
    parser.add_argument("--shard-size", type=int, default=600_000)
    args = parser.parse_args()

    wanted = {key(fen) for fen in load_fens(sorted(args.like.glob("*.npz")))}
    features, scores, mates = load_shards(list(args.source))
    fens = load_fens(list(args.source))
    keep = [i for i, fen in enumerate(fens) if key(fen) in wanted]
    print(f"{len(wanted):,} positions wanted, {len(keep):,} found in the source shards")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for index, start in enumerate(range(0, len(keep), args.shard_size)):
        rows = keep[start : start + args.shard_size]
        write_shard(
            args.out_dir / f"{args.prefix}_{index:04d}.npz",
            [(fens[i].decode(), int(scores[i]), int(mates[i]), [v for v in features[i] if v >= 0]) for i in rows],
        )
        print(f"{args.out_dir}/{args.prefix}_{index:04d}.npz: {len(rows):,} positions")


if __name__ == "__main__":
    main()
