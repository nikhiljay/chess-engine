"""Label candidate positions with Stockfish and write feature shards.

Each shard covers a fixed slice of the candidates file, so reruns skip shards
that already exist.

Usage:
  uv run python -m chess_nn.label --candidates data/candidates.txt --bench 2000 --depth 9
  uv run python -m chess_nn.label --candidates data/candidates.txt --depth 9
"""

import argparse
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import chess

from chess_nn.encoding import feature_indices
from chess_nn.harness import SHARDS_DIR
from chess_nn.shards import Row, write_shard

# Storage bound that fits int16. Scores are stored raw; training applies its own clamp.
SCORE_LIMIT_CP = 32000
SHARD_SIZE = 1_000_000


def parse_score(line: str) -> tuple[int, int] | None:
    """(centipawns, mate) from a UCI info line, from the side to move's view.

    mate is the signed moves-to-mate from the engine, or 0 when the score is in
    centipawns; mate scores report ±SCORE_LIMIT_CP as their centipawns.
    """
    tokens = line.split()
    if "score" not in tokens or "lowerbound" in tokens or "upperbound" in tokens:
        return None
    i = tokens.index("score")
    kind, value = tokens[i + 1], int(tokens[i + 2])
    if kind == "cp":
        return max(-SCORE_LIMIT_CP, min(SCORE_LIMIT_CP, value)), 0
    if kind == "mate":
        return (SCORE_LIMIT_CP if value > 0 else -SCORE_LIMIT_CP), value
    return None


def is_quiet_best_move(board: chess.Board, uci: str) -> bool:
    move = chess.Move.from_uci(uci)
    return not (board.is_capture(move) or move.promotion)


class Stockfish:
    """Minimal single-threaded UCI client."""

    def __init__(self, path: str = "stockfish", hash_mb: int = 16) -> None:
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        self._send("uci")
        self._read_until("uciok")
        self._send("setoption name Threads value 1")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("isready")
        self._read_until("readyok")

    def _send(self, command: str) -> None:
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def _readline(self) -> str:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("stockfish exited unexpectedly")
        return line

    def _read_until(self, prefix: str) -> str:
        while True:
            line = self._readline()
            if line.startswith(prefix):
                return line

    def analyse(self, fen: str, depth: int) -> tuple[tuple[int, int] | None, str]:
        """Return ((score_cp, mate) or None, best_move_uci) at a fixed depth."""
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        score = None
        while True:
            line = self._readline()
            if line.startswith("info"):
                parsed = parse_score(line)
                if parsed is not None:
                    score = parsed
            elif line.startswith("bestmove"):
                return score, line.split()[1]

    def close(self) -> None:
        self._send("quit")
        self.proc.wait(timeout=5)


def label_position(engine: Stockfish, fen: str, depth: int) -> Row | None:
    """Label one position; None if it has no score or its best move is not quiet."""
    board = chess.Board(fen)
    score, best = engine.analyse(fen, depth)
    if score is None or best == "(none)" or not is_quiet_best_move(board, best):
        return None
    score_cp, mate = score
    return fen, score_cp, mate, feature_indices(board)


_engine: Stockfish | None = None
_depth = 0


def _init_worker(stockfish_path: str, depth: int) -> None:
    global _engine, _depth
    _engine = Stockfish(stockfish_path)
    _depth = depth


def _label_worker(fen: str) -> Row | None:
    return label_position(_engine, fen, _depth)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=SHARDS_DIR)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--stockfish", default="stockfish")
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument("--limit", type=int, help="only use the first N candidates")
    parser.add_argument("--bench", type=int, help="label N candidates, report throughput, write nothing")
    args = parser.parse_args()

    with open(args.candidates) as f:
        fens = [line.strip() for line in f if line.strip()]
    if args.bench:
        fens = fens[: args.bench]
    elif args.limit:
        fens = fens[: args.limit]

    with Pool(args.workers, initializer=_init_worker, initargs=(args.stockfish, args.depth)) as pool:
        if args.bench:
            start = time.time()
            kept = sum(row is not None for row in pool.imap(_label_worker, fens, chunksize=16))
            elapsed = time.time() - start
            rate = len(fens) / elapsed
            kept_frac = kept / len(fens)
            hours_for_5m = 5_000_000 / (rate * kept_frac) / 3600
            print(
                f"depth={args.depth} workers={args.workers}: {rate:,.0f} positions/s, "
                f"{kept_frac:.1%} quiet, ~{hours_for_5m:.1f} h for 5M kept positions"
            )
            return

        for shard_idx, begin in enumerate(range(0, len(fens), args.shard_size)):
            path = args.out_dir / f"shard_{shard_idx:04d}.npz"
            chunk = fens[begin : begin + args.shard_size]
            if path.exists():
                print(f"{path.name}: exists, skipping", file=sys.stderr)
                continue
            rows = []
            start = time.time()
            for i, row in enumerate(pool.imap(_label_worker, chunk, chunksize=64), start=1):
                if row is not None:
                    rows.append(row)
                if i % 50_000 == 0:
                    rate = i / (time.time() - start)
                    eta_min = (len(chunk) - i) / rate / 60
                    print(
                        f"{path.name}: {i:,}/{len(chunk):,} labeled, {len(rows):,} kept "
                        f"({rate:,.0f}/s, shard ETA {eta_min:.0f} min)",
                        file=sys.stderr,
                    )
            write_shard(path, rows)
            print(f"{path.name}: wrote {len(rows):,} of {len(chunk):,} positions", file=sys.stderr)


if __name__ == "__main__":
    main()
