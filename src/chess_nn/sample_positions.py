"""Sample candidate training positions from a Lichess PGN archive.

Usage:
  uv run python -m chess_nn.sample_positions --pgn data/lichess.pgn.zst \
      --out data/candidates.txt --max-positions 8000000
"""

import argparse
import io
import json
import random
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import chess
import chess.pgn
import zstandard

from chess_nn.harness import OFFICIAL_OPENINGS, read_opening_fens

MIN_ELO = 1800
MIN_BASE_SECONDS = 180
SKIP_PLIES = 16
POSITIONS_PER_GAME = 6


def position_key(board: chess.Board) -> str:
    """FEN without move counters, normalized through python-chess."""
    return " ".join(board.fen().split()[:4])


def load_opening_keys(path: Path = OFFICIAL_OPENINGS) -> set[str]:
    return {position_key(chess.Board(fen)) for fen in read_opening_fens(path)}


def qualifies(headers: dict[str, str]) -> bool:
    try:
        white_elo = int(headers.get("WhiteElo", ""))
        black_elo = int(headers.get("BlackElo", ""))
    except ValueError:
        return False
    if min(white_elo, black_elo) < MIN_ELO:
        return False
    base = headers.get("TimeControl", "-").split("+")[0]
    if not base.isdigit() or int(base) < MIN_BASE_SECONDS:
        return False
    return headers.get("Termination") == "Normal"


class MainlineCollector(chess.pgn.BaseVisitor):
    """Collects mainline moves, skipping move parsing for non-qualifying games."""

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.moves: list[chess.Move] = []
        self.skipped = False

    def visit_header(self, tagname: str, tagvalue: str) -> None:
        self.headers[tagname] = tagvalue

    def end_headers(self):
        if not qualifies(self.headers):
            self.skipped = True
            return chess.pgn.SKIP
        return None

    def begin_variation(self):
        return chess.pgn.SKIP

    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        self.moves.append(move)

    def handle_error(self, error: Exception) -> None:
        self.skipped = True

    def result(self) -> "MainlineCollector":
        return self


def iter_games(path: Path) -> Iterator[MainlineCollector]:
    """Yield one collector per game in a .pgn or .pgn.zst file (also .zst.part)."""
    with open(path, "rb") as raw:
        if ".zst" in path.suffixes:
            # Lichess archives are multi-frame; stop only at end of file.
            stream = zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True)
        else:
            stream = raw
        text = io.TextIOWrapper(stream, encoding="utf-8", errors="replace")
        while True:
            collector = chess.pgn.read_game(text, Visitor=MainlineCollector)
            if collector is None:
                return
            yield collector


def sample_game(
    moves: list[chess.Move],
    rng: random.Random,
    opening_keys: set[str],
    skip_plies: int = SKIP_PLIES,
    per_game: int = POSITIONS_PER_GAME,
) -> list[chess.Board]:
    """Pick up to per_game quiet-candidate positions from a game's mainline."""
    board = chess.Board()
    eligible = []
    for ply, move in enumerate(moves, start=1):
        board.push(move)
        if ply <= skip_plies or board.is_check():
            continue
        if position_key(board) in opening_keys:
            continue
        eligible.append(board.copy(stack=False))
    rng.shuffle(eligible)
    chosen = []
    for candidate in eligible:
        if len(chosen) == per_game:
            break
        if any(candidate.generate_legal_moves()):
            chosen.append(candidate)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pgn", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-positions", type=int, default=8_000_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    opening_keys = load_opening_keys()
    seen: set[int] = set()
    games = qualifying = written = duplicates = 0
    start = time.time()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as out:
        for collector in iter_games(args.pgn):
            games += 1
            if games % 200_000 == 0:
                rate = games / (time.time() - start)
                print(
                    f"games={games:,} qualifying={qualifying:,} positions={written:,} "
                    f"dupes={duplicates:,} ({rate:,.0f} games/s)",
                    file=sys.stderr,
                )
            if collector.skipped or not collector.moves:
                continue
            qualifying += 1
            for board in sample_game(collector.moves, rng, opening_keys):
                key = hash(position_key(board))
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                out.write(board.fen() + "\n")
                written += 1
            if written >= args.max_positions:
                break

    stats = {
        "pgn": str(args.pgn),
        "games_read": games,
        "qualifying_games": qualifying,
        "positions": written,
        "duplicates_dropped": duplicates,
        "seconds": round(time.time() - start, 1),
    }
    args.out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
