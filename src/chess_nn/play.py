"""Play a game against a trained model in the terminal.

The net's moves come from the harness's play-move binary, so it searches
exactly as it does in compete (depth 1 + quiescence).

Usage:
  uv run python -m chess_nn.play runs/m1-h256/model.onnx --color black
"""

import argparse
import json
import random
import subprocess
from collections.abc import Callable
from pathlib import Path

import chess
import chess.pgn

from chess_nn.export import OnnxEvaluator
from chess_nn.harness import HARNESS_ROOT, PLAY_MOVE_BIN

HELP = "Enter moves like e4, Nf3, O-O or e2e4. Commands: moves, undo, fen, help, quit."


def render(board: chess.Board, perspective: chess.Color) -> str:
    ranks = range(7, -1, -1) if perspective == chess.WHITE else range(8)
    files = list(range(8)) if perspective == chess.WHITE else list(range(7, -1, -1))
    file_labels = "  " + " ".join(chess.FILE_NAMES[f] for f in files)
    lines = [file_labels]
    for rank in ranks:
        cells = []
        for file in files:
            piece = board.piece_at(chess.square(file, rank))
            cells.append(piece.unicode_symbol() if piece else "·")
        lines.append(f"{rank + 1} {' '.join(cells)} {rank + 1}")
    lines.append(file_labels)
    return "\n".join(lines)


def parse_human_move(board: chess.Board, text: str) -> chess.Move | None:
    """Parse SAN (e4, Nf3, O-O) or UCI (e2e4); None if unparseable or illegal."""
    text = text.strip()
    for parse in (board.parse_san, board.parse_uci):
        try:
            return parse(text)
        except ValueError:
            continue
    return None


def net_move(model: Path, board: chess.Board) -> chess.Move:
    if not PLAY_MOVE_BIN.exists():
        raise FileNotFoundError(f"{PLAY_MOVE_BIN} missing; run `cargo build --release -p cli` in {HARNESS_ROOT}")
    output = subprocess.run(
        [str(PLAY_MOVE_BIN), str(model.resolve()), board.fen()],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # The harness writes castling as king-takes-rook (e1h1); parse_uci normalizes it.
    return board.parse_uci(json.loads(output)["uci"])


def game_result(board: chess.Board) -> tuple[str, str] | None:
    """(result, reason) once the game is over, using the harness's end conditions."""
    if board.is_checkmate():
        return ("0-1" if board.turn == chess.WHITE else "1-0"), "checkmate"
    if board.is_stalemate():
        return "1/2-1/2", "stalemate"
    if board.is_repetition(3):
        return "1/2-1/2", "threefold repetition"
    if board.halfmove_clock >= 100:
        return "1/2-1/2", "fifty-move rule"
    if board.is_insufficient_material():
        return "1/2-1/2", "insufficient material"
    return None


def play(
    model: Path,
    human: chess.Color,
    start_fen: str | None = None,
    input_fn: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> chess.Board:
    board = chess.Board(start_fen) if start_fen else chess.Board()
    evaluator = OnnxEvaluator(model)
    out(HELP)
    result = None

    while True:
        over = game_result(board)
        if over:
            result, reason = over
            out("")
            out(render(board, human))
            out(f"Game over: {result} ({reason})")
            break

        if board.turn != human:
            move = net_move(model, board)
            san = board.san(move)
            board.push(move)
            # Side to move is now the human, so the eval is from your perspective.
            value = float(evaluator.evaluate([board])[0])
            out(f"Net plays {san}   (net's static eval for you: {value:+.2f})")
            continue

        out("")
        out(render(board, human))
        try:
            text = input_fn("Your move: ")
        except EOFError:
            break
        command = text.strip().lower()
        if command in ("quit", "exit"):
            break
        if command == "help":
            out(HELP)
        elif command == "moves":
            out(" ".join(sorted(board.san(m) for m in board.legal_moves)))
        elif command == "fen":
            out(board.fen())
        elif command == "undo":
            if len(board.move_stack) >= 2:
                board.pop()
                board.pop()
            else:
                out("Nothing to undo.")
        else:
            move = parse_human_move(board, text)
            if move is None:
                out(f"Not a legal move: {text.strip()!r}. Type 'moves' to list legal moves.")
            else:
                board.push(move)

    game = chess.pgn.Game.from_board(board)
    game.headers["White"] = "You" if human == chess.WHITE else model.parent.name
    game.headers["Black"] = "You" if human == chess.BLACK else model.parent.name
    game.headers["Result"] = result or "*"
    out("")
    out(str(game))
    return board


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("--color", choices=["white", "black", "random"], default="white")
    parser.add_argument("--fen", help="start from this position instead of the initial one")
    args = parser.parse_args()

    color = args.color if args.color != "random" else random.choice(["white", "black"])
    human = chess.WHITE if color == "white" else chess.BLACK
    print(f"You are {color}.")
    play(args.model, human, start_fen=args.fen)


if __name__ == "__main__":
    main()
