"""Compare evaluators against the teacher on the same leaves, sliced several neutral ways.

leaf_error reports error by teacher value. That is one slice, and it invites a story about
winning positions. This cuts the same leaves by properties that do not presuppose a cause —
game phase, material balance, whether captures or checks are pending, how cramped the side to
move is — and reports agreement with the depth-9 teacher in each, so a difference between two
architectures can be located before it is explained.

Correlation is reported alongside MAE because MAE rewards shrinking towards zero: finding 71
caught the weakest model looking best on MAE for exactly that reason.

Usage:
  uv run python -m chess_nn.arch_compare runs/teacher/leaves_d9_dev5.jsonl \\
      runs/jj6-h1024-s1000-cp20 runs/arch-cnn-c32b6 runs/arch-tf-d64l4-lr1e3
"""

import argparse
import json
from pathlib import Path

import chess
import numpy as np

from chess_nn.export import OnnxEvaluator


def slices_for(board: chess.Board) -> dict[str, str]:
    """Neutral descriptions of a position: no claim about what matters."""
    pieces = len(board.piece_map())
    material = sum(
        len(board.pieces(piece, colour)) * value
        for piece, value in ((chess.PAWN, 1), (chess.KNIGHT, 3), (chess.BISHOP, 3), (chess.ROOK, 5), (chess.QUEEN, 9))
        for colour in (chess.WHITE, chess.BLACK)
    )
    balance = sum(
        (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK))) * value
        for piece, value in ((chess.PAWN, 1), (chess.KNIGHT, 3), (chess.BISHOP, 3), (chess.ROOK, 5), (chess.QUEEN, 9))
    )
    if board.turn == chess.BLACK:
        balance = -balance
    return {
        "phase": "endgame" if material <= 20 else ("middlegame" if material <= 50 else "opening"),
        "pieces": "few (<=12)" if pieces <= 12 else ("many (>=24)" if pieces >= 24 else "medium"),
        "material": "even" if abs(balance) <= 1 else ("ahead" if balance > 1 else "behind"),
        "captures": "captures pending" if any(board.generate_legal_captures()) else "no captures",
        "mobility": "cramped (<=20 moves)" if board.legal_moves.count() <= 20 else "free (>20 moves)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("leaves", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--sample", type=int, default=8000)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.leaves.read_text().splitlines() if line.strip()][: args.sample]
    boards = [chess.Board(r["fen"]) for r in rows]
    teacher = np.array([r["teacher"] for r in rows])
    tags = [slices_for(board) for board in boards]
    print(f"{len(rows):,} leaves, teacher from depth 9\n")

    values = {run.name: np.asarray(OnnxEvaluator(run / "model.onnx").evaluate(boards)) for run in args.runs}
    names = list(values)
    dimensions = ["phase", "pieces", "material", "captures", "mobility"]
    for dimension in dimensions:
        options = sorted({tag[dimension] for tag in tags})
        print(f"--- {dimension}")
        header = "".join(f"{name[:22]:>24s}" for name in names)
        print(f"{'slice':22s}{'n':>7s}{header}")
        for option in options:
            mask = np.array([tag[dimension] == option for tag in tags])
            if mask.sum() < 50:
                continue
            cells = ""
            for name in names:
                predicted, truth = values[name][mask], teacher[mask]
                correlation = float(np.corrcoef(predicted, truth)[0, 1]) if predicted.std() > 1e-9 else float("nan")
                cells += f"{f'r {correlation:.3f} MAE {np.abs(predicted - truth).mean():.2f}':>24s}"
            print(f"{option:22s}{int(mask.sum()):>7d}{cells}")
        print()


if __name__ == "__main__":
    main()
