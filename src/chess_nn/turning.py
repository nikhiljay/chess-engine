"""Lost games: large evaluation drops versus moves that turn a defensible position into a losing one.

A drop of several pawns is not always what lost a game: it can happen in a
position that was already lost, or leave a winning side still winning. A
turning move is one from a defensible position (Stockfish's best, with the game
history, at least DEFENSIBLE_CP) to a losing one (the net's move at most
-WIN_CP). This report counts both kinds in the net's lost games, from existing
chess_nn.diagnose output, so training analysis can target the turning moves.

Each diagnosis directory is given with the game records it was built from, in
the same order (diagnose numbers the games it kept after dropping wins).

Usage:
  uv run python -m chess_nn.turning \\
      data/diagnosis_h1024_L2=data/games_h1024_L2.jsonl \\
      data/diagnosis_b_h1024=data/games_b_h1024_L2.jsonl,data/games_b_h1024_L3.jsonl
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import chess

from chess_nn.diagnose import BLUNDER_CP, WIN_CP
from chess_nn.mate_in_one import unseen_mate
from chess_nn.rank_pairs import DEFENSIBLE_CP, is_decisive


def net_won(game: dict) -> bool:
    return game["result"] == ("1-0" if game["net_white"] else "0-1")


def net_lost(game: dict) -> bool:
    return game["result"] == ("0-1" if game["net_white"] else "1-0")


def loss_breakdown(decisions: list[dict]) -> dict:
    """Per lost game: turning moves versus large drops that did not turn the game."""
    ordered = sorted(decisions, key=lambda d: d["ply"])
    turning = [d for d in ordered if is_decisive(d)]
    large = [d for d in ordered if d["best_cp"] - d["played_cp"] >= BLUNDER_CP]
    return {
        "turning_moves": len(turning),
        "first_turning": turning[0] if turning else None,
        "large_drops": len(large),
        "large_drops_not_turning": sum(1 for d in large if not is_decisive(d)),
        "ever_defensible": any(d["best_cp"] >= DEFENSIBLE_CP for d in ordered),
    }


def load_sets(specs: list[str]) -> list[tuple[dict, list[dict]]]:
    """(game, its decisions) for every game each diagnosis kept."""
    out = []
    for spec in specs:
        directory, files = spec.split("=", 1)
        games = [json.loads(line) for f in files.split(",") for line in Path(f).read_text().splitlines() if line.strip()]
        kept = [g for g in games if not net_won(g)]
        by_game = defaultdict(list)
        for line in (Path(directory) / "decisions.jsonl").read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                by_game[d["game"]].append(d)
        out += [(game, by_game[i]) for i, game in enumerate(kept)]
    return out


def report(sets: list[tuple[dict, list[dict]]]) -> dict:
    losses = [(g, loss_breakdown(ds)) for g, ds in sets if net_lost(g)]
    with_turning = [b for _, b in losses if b["turning_moves"]]
    firsts = [b["first_turning"] for b in with_turning]
    unseen = 0
    for d in firsts:
        board = chess.Board(d["fen"])
        if unseen_mate(board, chess.Move.from_uci(d["played"])) is not None:
            unseen += 1
    return {
        "lost_games": len(losses),
        "with_a_turning_move": len(with_turning),
        "large_drop_but_no_turning_move": sum(1 for _, b in losses if b["large_drops"] and not b["turning_moves"]),
        "never_defensible": sum(1 for _, b in losses if not b["ever_defensible"]),
        "turning_moves_per_game_median": statistics.median(b["turning_moves"] for b in with_turning) if with_turning else None,
        "first_turning_ply_median": statistics.median(d["ply"] for d in firsts) if firsts else None,
        "first_turning_drop_cp_median": statistics.median(d["best_cp"] - d["played_cp"] for d in firsts) if firsts else None,
        "first_turning_allows_unseen_mate": unseen,
        "large_drops_total": sum(b["large_drops"] for _, b in losses),
        "large_drops_not_turning_total": sum(b["large_drops_not_turning"] for _, b in losses),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sets", nargs="+", help="DIAGNOSIS_DIR=games1.jsonl[,games2.jsonl]")
    args = parser.parse_args()
    print(json.dumps(report(load_sets(args.sets)), indent=2))


if __name__ == "__main__":
    main()
