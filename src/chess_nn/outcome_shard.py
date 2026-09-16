"""Training shards labeled with the game's actual result, not Stockfish's opinion.

Every experiment so far distils a Stockfish score, and findings 73-74 showed why that target
misleads: it is measured on a position the harness search will not resolve the same way, and
the most faithful distillation of a static evaluation played 12 points worse. The thing we
actually measure is whether the net beats these baseline bots, and an outcome label optimises
that directly — including the search's own quirks, the opponent's habits, and the fact that a
repetition draw scores half a point.

One bit per game is a noisy target, so this wants many games; datagen produces them cheaply
(about 400 per minute). Positions where the side to move is in check are skipped, since
quiescence never takes a stand-pat value there.

Targets are written as centipawns so the usual WDL and centipawn losses apply unchanged:
a win for the side to move is +win_cp, a loss is -win_cp, a draw is 0.

Usage:
  uv run python -m chess_nn.outcome_shard data/outcome_games_L2.jsonl \\
      --out-dir data/shards_outcome --prefix shard_outcome_a
"""

import argparse
import json
from pathlib import Path

import chess

from chess_nn.encoding import feature_indices
from chess_nn.shards import Row, write_shard


def rows_from_game(game: dict, win_cp: int, skip_plies: int) -> list[Row]:
    """One row per position after `skip_plies`, labeled from the side to move's view."""
    board = chess.Board(game["start_fen"])
    result = game["result"]
    rows: list[Row] = []
    for ply, uci in enumerate(game["moves"]):
        if ply >= skip_plies and not board.is_check():
            if result == "1/2-1/2":
                score = 0
            else:
                white_won = result == "1-0"
                score = win_cp if white_won == (board.turn == chess.WHITE) else -win_cp
            rows.append((board.fen(), score, 0, feature_indices(board)))
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            break
        board.push(move)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("games", type=Path, nargs="+", help="datagen --record JSONL files")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="shard_outcome")
    parser.add_argument("--win-cp", type=int, default=2000, help="target for a win from the side to move")
    parser.add_argument("--skip-plies", type=int, default=8, help="ignore the random opening plies")
    parser.add_argument("--shard-size", type=int, default=600_000)
    args = parser.parse_args()

    rows: list[Row] = []
    seen: set[str] = set()
    results: dict[str, int] = {}
    for path in args.games:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            game = json.loads(line)
            results[game["result"]] = results.get(game["result"], 0) + 1
            for row in rows_from_game(game, args.win_cp, args.skip_plies):
                key = " ".join(row[0].split()[:4])
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
    print(f"{len(rows):,} positions from {sum(results.values()):,} games {results}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for index, start in enumerate(range(0, len(rows), args.shard_size)):
        chunk = rows[start : start + args.shard_size]
        path = args.out_dir / f"{args.prefix}_{index:04d}.npz"
        write_shard(path, chunk)
        print(f"{path}: {len(chunk):,} positions")


if __name__ == "__main__":
    main()
