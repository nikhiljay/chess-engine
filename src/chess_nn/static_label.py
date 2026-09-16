"""Label positions with Stockfish's static evaluation, and write training shards.

Findings 67-73 point here. A leaf value inside this harness must say what the position is
worth as it stands: the search resolves captures itself, so a depth-9 target makes the net
count them twice (finding 73, where unfiltered leaf targets cost 21 points at L2). Stockfish's
static evaluation is the function that plays that role inside Stockfish's own search, and as a
leaf function in our teacher test it scored 68.8% against our net's 56.2%.

It is also a better learning target than a search score: deterministic, so the labels carry no
search noise, defined for every quiet position with no filtering, and about five times cheaper
to produce than depth 9.

Positions where the side to move is in check are skipped: quiescence never takes a stand-pat
value there, so the evaluator is never asked about them.

Usage:
  uv run python -m chess_nn.static_label --candidates data/leaf_candidates_a.txt \\
      --out-dir data/shards_static --prefix shard_static_leaf
"""

import argparse
import subprocess
from pathlib import Path

import chess

from chess_nn.encoding import feature_indices
from chess_nn.shards import Row, write_shard

SCORE_LIMIT_CP = 2500
_engine: subprocess.Popen | None = None


def _start(path: str) -> None:
    global _engine
    _engine = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    for command in ("uci", "setoption name Threads value 1", "setoption name Hash value 16", "isready"):
        _engine.stdin.write(command + "\n")
    _engine.stdin.flush()
    while not _engine.stdout.readline().startswith("readyok"):
        pass


def _static_eval(fen: str) -> Row | None:
    """(fen, centipawns from the side to move, 0, features), or None when unusable."""
    board = chess.Board(fen)
    if board.is_check() or not board.is_valid():
        return None
    _engine.stdin.write(f"position fen {fen}\neval\n")
    _engine.stdin.flush()
    while True:
        line = _engine.stdout.readline()
        if not line:
            return None
        if line.startswith("Final evaluation"):
            token = line.split()[2]
            if token.startswith("none"):
                return None
            white_side = float(token)
            pawns = white_side if board.turn == chess.WHITE else -white_side
            score = max(-SCORE_LIMIT_CP, min(SCORE_LIMIT_CP, int(round(pawns * 100))))
            return fen, score, 0, feature_indices(board)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="shard_static")
    parser.add_argument("--shard-size", type=int, default=600_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--stockfish", default="stockfish")
    args = parser.parse_args()

    fens = [line.strip() for line in args.candidates.read_text().splitlines() if line.strip()]
    if args.limit:
        fens = fens[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from multiprocessing import Pool

    kept: list[Row] = []
    written = 0
    with Pool(args.workers, initializer=_start, initargs=(args.stockfish,)) as pool:
        for index, row in enumerate(pool.imap(_static_eval, fens, chunksize=256), start=1):
            if row is not None:
                kept.append(row)
            if len(kept) >= args.shard_size:
                path = args.out_dir / f"{args.prefix}_{written:04d}.npz"
                write_shard(path, kept)
                print(f"{path}: {len(kept):,} positions ({index:,}/{len(fens):,} scanned)", flush=True)
                kept, written = [], written + 1
    if kept:
        path = args.out_dir / f"{args.prefix}_{written:04d}.npz"
        write_shard(path, kept)
        print(f"{path}: {len(kept):,} positions", flush=True)


if __name__ == "__main__":
    main()
