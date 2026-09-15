"""Build a dev opening set disjoint from the official harness openings.

Takes the positions after a fixed number of plies from qualifying Lichess
games, drops any game whose opening passes through an official opening
position, and samples from the most common remaining positions.

Usage:
  uv run python -m chess_nn.dev_openings --pgn data/lichess.pgn.zst
"""

import argparse
import random
from collections import Counter
from pathlib import Path

import chess

from chess_nn.harness import DEV_OPENINGS, read_opening_fens
from chess_nn.sample_positions import iter_games, load_opening_keys, position_key


def build_dev_openings(
    pgn: Path,
    count: int = 50,
    ply: int = 6,
    games: int = 200_000,
    pool: int = 300,
    seed: int = 0,
    exclude: list[Path] | None = None,
) -> list[str]:
    official = load_opening_keys()
    for path in exclude or []:
        official |= {position_key(chess.Board(fen)) for fen in read_opening_fens(path)}
    counts: Counter[str] = Counter()
    examples: dict[str, str] = {}
    used = 0
    for collector in iter_games(pgn):
        if collector.skipped or len(collector.moves) < ply:
            continue
        board = chess.Board()
        passes_official = False
        for move in collector.moves[:ply]:
            board.push(move)
            if position_key(board) in official:
                passes_official = True
                break
        if passes_official or board.is_check():
            continue
        key = position_key(board)
        counts[key] += 1
        examples.setdefault(key, board.fen())
        used += 1
        if used >= games:
            break
    top = [key for key, _ in counts.most_common(pool)]
    chosen = random.Random(seed).sample(top, min(count, len(top)))
    return [examples[key] for key in chosen]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pgn", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEV_OPENINGS)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ply", type=int, default=6)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[], help="opening files whose positions a line may not pass through")
    args = parser.parse_args()

    fens = build_dev_openings(args.pgn, count=args.count, ply=args.ply, seed=args.seed, exclude=args.exclude)
    header = [
        f"# Openings: {len(fens)} common Lichess positions after {args.ply} plies ({args.pgn.name}, seed {args.seed}),",
        "# excluding any line that passes through an official opening" + (f" or {', '.join(p.name for p in args.exclude)}" if args.exclude else "") + ".",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(header + fens) + "\n")
    print(f"Wrote {len(fens)} dev openings to {args.out}")


if __name__ == "__main__":
    main()
