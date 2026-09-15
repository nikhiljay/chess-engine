"""Diagnose the net's recorded games with Stockfish, using the game history.

Stockfish is given each position as `position fen <start> moves ...`, so it
knows the game so far: a shuffle back into a position that already occurred
scores as a draw, even when the position on its own looks winning. For every
move the net made, the report scores Stockfish's best move and the net's move
at a fixed depth, then counts:

  - repetition draws where the net really had a win (best move >= WIN_CP),
  - the move where that win slipped (best still winning, the net's move drawish)
    and whether that move repeated a position,
  - the biggest drop in each lost game.

Every decision goes to decisions.jsonl — the material for a move-ranking
objective: the position, Stockfish's best move and the net's move, both scored.

Usage:
  datagen/target/release/datagen runs/<run>/model.onnx --out /dev/null --record data/games_L2.jsonl \
      --games 400 --level 2 --random-move-prob 0 --sample-prob 0
  uv run python -m chess_nn.diagnose data/games_L2.jsonl --out data/diagnosis_L2 --depth 14
"""

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import chess

from chess_nn.label import Stockfish, parse_score

WIN_CP = 300  # the net's side is clearly winning
DRAWISH_CP = 100  # a move that leaves at most this much has given the win away
BLUNDER_CP = 300  # a move this much worse than the best is a blunder
MATE_CP = 10_000


def to_cp(score: tuple[int, int] | None) -> int:
    """Stockfish (cp, mate) → a single number, mates beyond any centipawn score."""
    if score is None:
        return 0
    cp, mate = score
    if mate:
        return MATE_CP - abs(mate) if mate > 0 else -MATE_CP + abs(mate)
    return cp


def search(engine: Stockfish, position: str, depth: int, move: str | None = None) -> tuple[int, str]:
    """(score from the side to move's view, best move) for a `position ...` command."""
    engine._send(position)
    engine._send(f"go depth {depth}" + (f" searchmoves {move}" if move else ""))
    score = None
    while True:
        line = engine._readline()
        if line.startswith("info"):
            parsed = parse_score(line)
            if parsed is not None:
                score = parsed
        elif line.startswith("bestmove"):
            return to_cp(score), line.split()[1]


def summarize_game(game: dict, decisions: list[dict]) -> dict:
    """Game-level findings from its scored decisions (pure; no engine)."""
    winning = [d for d in decisions if d["best_cp"] >= WIN_CP]
    slip = next((d for d in winning if d["played_cp"] <= DRAWISH_CP), None)
    # Only moves from positions not already lost can be the blunder that lost the game.
    playable = [d for d in decisions if d["best_cp"] > -WIN_CP]
    worst = max(playable, key=lambda d: d["best_cp"] - d["played_cp"], default=None)
    blunder = worst if worst and worst["best_cp"] - worst["played_cp"] >= BLUNDER_CP else None
    net_color_won = game["result"] == ("1-0" if game["net_white"] else "0-1")
    net_color_lost = game["result"] == ("0-1" if game["net_white"] else "1-0")
    return {
        "end": game["end"],
        "level": game["level"],
        "outcome": "win" if net_color_won else "loss" if net_color_lost else "draw",
        "plies": len(game["moves"]),
        "had_win": bool(winning),
        "first_win_ply": winning[0]["ply"] if winning else None,
        "best_eval_cp": max((d["best_cp"] for d in decisions), default=None),
        "slip": slip,
        "blunder": blunder,
        "net_moves": len(decisions),
        "agree": sum(d["played"] == d["best"] for d in decisions),
    }


_engine: Stockfish | None = None
_depth = 14


def _init(depth: int) -> None:
    global _engine, _depth
    _engine = Stockfish(hash_mb=64)
    _depth = depth


def diagnose_game(args: tuple[int, dict]) -> tuple[dict, list[dict]]:
    index, game = args
    board = chess.Board(game["start_fen"])
    net_color = chess.WHITE if game["net_white"] else chess.BLACK
    seen = Counter([board._transposition_key()])
    history: list[str] = []
    decisions = []
    for ply, raw in enumerate(game["moves"]):
        move = board.parse_uci(raw)  # the harness writes castling as king-takes-rook
        uci = move.uci()
        if board.turn == net_color:
            position = f"position fen {game['start_fen']}" + (f" moves {' '.join(history)}" if history else "")
            best_cp, best = search(_engine, position, _depth)
            played_cp = best_cp if best == uci else search(_engine, position, _depth, uci)[0]
            board.push(move)
            repeats = seen[board._transposition_key()] >= 1
            board.pop()
            decisions.append({
                "game": index,
                "ply": ply,
                "fen": board.fen(),
                "history": list(history),
                "start_fen": game["start_fen"],
                "best": best,
                "best_cp": best_cp,
                "played": uci,
                "played_cp": played_cp,
                "repeats": repeats,
            })
        board.push(move)
        seen[board._transposition_key()] += 1
        history.append(uci)
    return summarize_game(game, decisions), decisions


def report(summaries: list[dict]) -> dict:
    by_end = Counter(s["end"] for s in summaries)
    outcomes = Counter(s["outcome"] for s in summaries)
    repetition = [s for s in summaries if s["end"] == "repetition"]
    rep_won = [s for s in repetition if s["had_win"]]
    rep_slipped = [s for s in rep_won if s["slip"]]
    losses = [s for s in summaries if s["outcome"] == "loss"]
    lost_after_win = [s for s in losses if s["had_win"]]
    moves = sum(s["net_moves"] for s in summaries)
    return {
        "games": len(summaries),
        "outcomes": dict(outcomes),
        "end_reasons": dict(by_end),
        "net_move_agreement": round(sum(s["agree"] for s in summaries) / max(moves, 1), 3),
        "repetition": {
            "games": len(repetition),
            "had_win": len(rep_won),
            "slipped": len(rep_slipped),
            "slip_move_repeats_position": sum(s["slip"]["repeats"] for s in rep_slipped),
            "median_plies_slip_to_end": statistics.median(s["plies"] - s["slip"]["ply"] for s in rep_slipped) if rep_slipped else None,
            "median_best_eval_cp": statistics.median(s["best_eval_cp"] for s in rep_won) if rep_won else None,
        },
        "draws_other_with_win": sum(1 for s in summaries if s["outcome"] == "draw" and s["end"] != "repetition" and s["had_win"]),
        "losses": {
            "games": len(losses),
            "had_win_first": len(lost_after_win),
            "with_blunder": sum(1 for s in losses if s["blunder"]),
            "median_blunder_ply": statistics.median(s["blunder"]["ply"] for s in losses if s["blunder"]) if any(s["blunder"] for s in losses) else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("games", type=Path, nargs="+", help="JSONL game records from datagen --record")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=14)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--only", choices=["all", "not-won"], default="not-won", help="skip games the net won")
    args = parser.parse_args()

    games = [json.loads(line) for path in args.games for line in path.read_text().splitlines() if line.strip()]
    if args.only == "not-won":
        games = [g for g in games if g["result"] != ("1-0" if g["net_white"] else "0-1")]
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    summaries = []
    with open(args.out / "decisions.jsonl", "w") as decisions_file, open(args.out / "games.jsonl", "w") as games_file:
        with Pool(args.workers, initializer=_init, initargs=(args.depth,)) as pool:
            for i, (summary, decisions) in enumerate(pool.imap_unordered(diagnose_game, list(enumerate(games))), start=1):
                summaries.append(summary)
                games_file.write(json.dumps(summary) + "\n")
                for d in decisions:
                    decisions_file.write(json.dumps(d) + "\n")
                if i % 25 == 0:
                    print(f"{i}/{len(games)} games ({time.time() - start:.0f}s)", file=sys.stderr)
    result = report(summaries)
    result["depth"] = args.depth
    (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
