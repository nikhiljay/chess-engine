"""Stockfish playouts of positions the net failed to convert, labeled by distance to mate.

The net reaches winning positions in most games it does not win, then shuffles
into a repetition or blunders (finding 38). A depth-1 search can only convert if
the evaluation rises as mate approaches, but every label source so far scores a
won position the same whether mate is near or the side to move is shuffling:
depth-9 Stockfish scores barely separate progress from shuffles (findings 9,
24), and the default recipe clamps every mate to the same ±25 pawns.

This takes clearly winning positions (best_cp >= --min-cp) from diagnosed games
the net did not win, replays each game's history into Stockfish so it avoids
repetitions, and lets Stockfish play both sides at a fixed depth until the game
ends. For playouts that end in mate, every position on the way is labeled from
the side to move's view, positive for the side that mates:

    value = min(clamp, max(stockfish cp for the mating side, 0) + slope * max(0, horizon - plies to mate))

so positions far from mate keep Stockfish's own score and the last `horizon`
plies rise toward the clamp. Positions whose playout move is a capture or
promotion are dropped (chess_nn.label's quiet-best-move filter), as are
positions that occur in --exclude diagnoses; playouts that do not end in mate
contribute nothing.

Usage:
  uv run python -m chess_nn.playouts data/diagnosis_h1024_L2 data/diagnosis_b_h1024 \\
      --exclude data/diagnosis_heldout_h1024 data/diagnosis_fresh_h1024 \\
      --out data/shards_playouts/shard_playouts_a_0000.npz
"""

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import chess

from chess_nn.decision_data import load_decisions, position_key
from chess_nn.diagnose import search
from chess_nn.encoding import feature_indices
from chess_nn.label import Stockfish
from chess_nn.shards import Row, write_shard

CLAMP_CP = 2500


def distance_value(mating_side_cp: int, plies_to_mate: int, slope_cp: float, horizon_plies: int, clamp_cp: int = CLAMP_CP) -> int:
    """The label's magnitude for a position `plies_to_mate` plies before the mate."""
    bonus = slope_cp * max(0, horizon_plies - plies_to_mate)
    return int(min(clamp_cp, max(mating_side_cp, 0) + bonus))


def playout(choose: Callable[[str], tuple[int, str]], start_fen: str, history: list[str], max_plies: int) -> dict:
    """Play from start_fen + history with `choose(position command) -> (cp, uci)` until the game ends.

    Returns the result ('mate', 'draw' or 'unfinished'), the mating colour, and one step per ply:
    (fen, cp for the side to move, whether the move played is a capture or promotion, side to move).
    """
    board = chess.Board(start_fen)
    moves = list(history)
    for uci in moves:
        board.push_uci(uci)
    steps = []
    while True:
        if board.is_checkmate():
            return {"result": "mate", "winner": not board.turn, "steps": steps}
        if board.is_stalemate() or board.is_insufficient_material() or board.halfmove_clock >= 100 or board.is_repetition(3):
            return {"result": "draw", "winner": None, "steps": steps}
        if len(steps) >= max_plies:
            return {"result": "unfinished", "winner": None, "steps": steps}
        cp, best = choose(f"position fen {start_fen}" + (f" moves {' '.join(moves)}" if moves else ""))
        move = board.parse_uci(best)
        steps.append((board.fen(), cp, board.is_capture(move) or move.promotion is not None, board.turn))
        board.push(move)
        moves.append(best)


def playout_rows(game: dict, slope_cp: float, horizon_plies: int) -> list[Row]:
    """Training rows for a playout that ended in mate: quiet positions with distance labels."""
    if game["result"] != "mate":
        return []
    rows = []
    n = len(game["steps"])
    for i, (fen, cp, loud, turn) in enumerate(game["steps"]):
        if loud:
            continue
        mating_side_to_move = turn == game["winner"]
        value = distance_value(cp if mating_side_to_move else -cp, n - i, slope_cp, horizon_plies)
        rows.append((fen, value if mating_side_to_move else -value, 0, feature_indices(chess.Board(fen))))
    return rows


def pick_decisions(directories: list[Path], min_cp: int, per_game: int, seed: int) -> list[dict]:
    """Up to per_game clearly winning decisions from each diagnosed game, seeded."""
    rng = random.Random(seed)
    picked = []
    for directory in directories:
        by_game = defaultdict(list)
        for d in load_decisions(directory):
            if d["best_cp"] >= min_cp:
                by_game[d["game"]].append(d)
        for game in sorted(by_game):
            candidates = by_game[game]
            picked += rng.sample(candidates, min(per_game, len(candidates)))
    return picked


_engine: Stockfish | None = None
_args: tuple[int, int, float, int] = (8, 300, 15.0, 100)


def _init(depth: int, max_plies: int, slope_cp: float, horizon_plies: int) -> None:
    global _engine, _args
    _engine = Stockfish(hash_mb=64)
    _args = (depth, max_plies, slope_cp, horizon_plies)


def _run(decision: dict) -> tuple[str, int, list[Row]]:
    depth, max_plies, slope_cp, horizon_plies = _args
    game = playout(lambda position: search(_engine, position, depth), decision["start_fen"], decision["history"], max_plies)
    return game["result"], len(game["steps"]), playout_rows(game, slope_cp, horizon_plies)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("diagnoses", type=Path, nargs="+", help="training-source chess_nn.diagnose directories")
    parser.add_argument("--exclude", type=Path, nargs="*", default=[], help="held-out diagnoses whose positions are dropped")
    parser.add_argument("--out", type=Path, required=True, help="shard path")
    parser.add_argument("--min-cp", type=int, default=500)
    parser.add_argument("--per-game", type=int, default=2)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--slope-cp", type=float, default=15.0, help="label rise per ply closer to mate")
    parser.add_argument("--horizon-plies", type=int, default=100, help="plies before mate where the rise starts")
    parser.add_argument("--limit", type=int, help="only this many playouts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    excluded = {position_key(chess.Board(d["fen"])) for directory in args.exclude for d in load_decisions(directory)}
    decisions = pick_decisions(args.diagnoses, args.min_cp, args.per_game, args.seed)
    random.Random(args.seed).shuffle(decisions)
    decisions = decisions[: args.limit]
    start = time.time()
    results, lengths, rows, seen, dropped = Counter(), [], [], set(), 0
    init = (args.depth, args.max_plies, args.slope_cp, args.horizon_plies)
    with Pool(args.workers, initializer=_init, initargs=init) as pool:
        for i, (result, plies, game_rows) in enumerate(pool.imap_unordered(_run, decisions, chunksize=1), start=1):
            results[result] += 1
            lengths.append(plies)
            for row in game_rows:
                key = position_key(chess.Board(row[0]))
                if key in excluded:
                    dropped += 1
                elif key not in seen:
                    seen.add(key)
                    rows.append(row)
            if i % 500 == 0:
                print(f"{i:,}/{len(decisions):,} playouts, {len(rows):,} positions, {time.time() - start:.0f}s", file=sys.stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_shard(args.out, rows)
    summary = {
        "playouts": len(decisions),
        "results": dict(results),
        "median_plies": statistics.median(lengths) if lengths else None,
        "positions": len(rows),
        "held_out_duplicates_dropped": dropped,
        "seconds": round(time.time() - start),
        **{k: v for k, v in vars(args).items() if k not in ("diagnoses", "exclude", "out")},
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), file=sys.stderr)


if __name__ == "__main__":
    main()
