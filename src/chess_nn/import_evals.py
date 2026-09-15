"""Convert the Lichess evaluation database into training shards.

Each line of lichess_db_eval.jsonl holds a position (FEN without move counters)
and one or more Stockfish analyses. The deepest analysis's first principal
variation gives the score and best move, and positions are filtered like
chess_nn.label: not in check, best move legal and quiet, and not an official
opening position. The file reports scores from White's view; shards store them
from the side to move's view, like Stockfish's UCI output.

Sampling is seeded, so a rerun keeps the same positions and skips shards that
already exist (it still re-reads the file up to the first missing shard).

Usage:
  uv run python -m chess_nn.import_evals --evals data/lichess_db_eval.jsonl.zst \
      --sample-prob 0.1 --max-positions 10000000
"""

import argparse
import io
import json
import random
import sys
import time
from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path

import chess
import zstandard

from chess_nn.encoding import feature_indices
from chess_nn.harness import SHARDS_DIR
from chess_nn.label import SCORE_LIMIT_CP, SHARD_SIZE, is_quiet_best_move
from chess_nn.sample_positions import load_opening_keys, position_key
from chess_nn.shards import Row, write_shard


def best_analysis(record: dict) -> tuple[int, int, int, str] | None:
    """(depth, cp, mate, best move) from the deepest analysis, White's view.

    mate is the signed moves-to-mate (0 for centipawn scores); cp is 0 for mates.
    """
    evals = record.get("evals") or []
    if not evals:
        return None
    deepest = max(evals, key=lambda e: (e.get("depth", 0), e.get("knodes", 0)))
    pvs = deepest.get("pvs") or []
    if not pvs:
        return None
    pv = pvs[0]
    moves = pv.get("line", "").split()
    if not moves:
        return None
    if "mate" in pv:
        if not pv["mate"]:
            return None
        return deepest.get("depth", 0), 0, int(pv["mate"]), moves[0]
    if "cp" not in pv:
        return None
    return deepest.get("depth", 0), int(pv["cp"]), 0, moves[0]


def row_from_record(record: dict, min_depth: int, opening_keys: set[str]) -> Row | None:
    """A shard row with the score from the side to move's view, or None if filtered out."""
    analysis = best_analysis(record)
    if analysis is None:
        return None
    depth, cp, mate, best = analysis
    if depth < min_depth:
        return None
    try:
        board = chess.Board(record["fen"])
        move = chess.Move.from_uci(best)
    except (KeyError, ValueError):
        return None
    # Analysis boards can be editor setups (e.g. more than 16 pieces a side) that the encoding cannot hold.
    if not board.is_valid() or board.is_check() or not board.is_legal(move) or not is_quiet_best_move(board, best):
        return None
    if position_key(board) in opening_keys:
        return None
    sign = 1 if board.turn == chess.WHITE else -1
    if mate:
        mate *= sign
        score = SCORE_LIMIT_CP if mate > 0 else -SCORE_LIMIT_CP
    else:
        score = max(-SCORE_LIMIT_CP, min(SCORE_LIMIT_CP, sign * cp))
    return board.fen(), score, mate, feature_indices(board)


def iter_lines(path: Path) -> Iterator[str]:
    """Lines of a .jsonl or multi-frame .jsonl.zst file."""
    with open(path, "rb") as raw:
        if ".zst" in path.suffixes:
            stream = zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True)
        else:
            stream = raw
        yield from io.TextIOWrapper(stream, encoding="utf-8")


_min_depth = 0
_opening_keys: set[str] = set()


def _init_worker(min_depth: int, opening_keys: set[str]) -> None:
    global _min_depth, _opening_keys
    _min_depth, _opening_keys = min_depth, opening_keys


def _convert_line(line: str) -> Row | None:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    return row_from_record(record, _min_depth, _opening_keys)


def convert(
    evals: Path,
    out_dir: Path,
    prefix: str = "shard_evaldb",
    sample_prob: float = 1.0,
    max_positions: int | None = None,
    min_depth: int = 0,
    shard_size: int = SHARD_SIZE,
    workers: int = 8,
    seed: int = 0,
) -> dict:
    """Write shards of kept positions; returns counts. workers=0 converts in-process."""
    rng = random.Random(seed)
    sampled_lines = (line for line in iter_lines(evals) if rng.random() < sample_prob)
    init_args = (min_depth, load_opening_keys())
    stats = {"sampled": 0, "kept": 0, "shards_written": 0}
    shard_idx, rows = 0, []
    start = time.time()

    def flush() -> None:
        nonlocal shard_idx, rows
        path = out_dir / f"{prefix}_{shard_idx:04d}.npz"
        if path.exists():
            print(f"{path.name}: exists, skipping", file=sys.stderr)
        else:
            write_shard(path, rows)
            stats["shards_written"] += 1
            print(f"{path.name}: wrote {len(rows):,} positions", file=sys.stderr)
        shard_idx, rows = shard_idx + 1, []

    if workers > 0:
        pool = Pool(workers, initializer=_init_worker, initargs=init_args)
        results = pool.imap(_convert_line, sampled_lines, chunksize=512)
    else:
        pool = None
        _init_worker(*init_args)
        results = map(_convert_line, sampled_lines)
    try:
        for row in results:
            stats["sampled"] += 1
            if stats["sampled"] % 1_000_000 == 0:
                rate = stats["sampled"] / (time.time() - start)
                print(f"sampled={stats['sampled']:,} kept={stats['kept']:,} ({rate:,.0f}/s)", file=sys.stderr)
            if row is None:
                continue
            rows.append(row)
            stats["kept"] += 1
            if len(rows) == shard_size:
                flush()
            if max_positions and stats["kept"] >= max_positions:
                break
    finally:
        if pool is not None:
            pool.terminate()
    if rows:
        flush()
    stats["seconds"] = round(time.time() - start, 1)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evals", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=SHARDS_DIR)
    parser.add_argument("--prefix", default="shard_evaldb", help="shard file prefix; keep it starting with 'shard'")
    parser.add_argument("--sample-prob", type=float, default=1.0, help="keep each line with this probability")
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--min-depth", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stats = convert(
        args.evals,
        args.out_dir,
        prefix=args.prefix,
        sample_prob=args.sample_prob,
        max_positions=args.max_positions,
        min_depth=args.min_depth,
        shard_size=args.shard_size,
        workers=args.workers,
        seed=args.seed,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
