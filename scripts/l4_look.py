"""Show how the Level 4 games actually end.

Both are drawn by repetition, so the question is which side stood better when the shuffling began.
If we are ahead, Level 4 is a conversion problem; if we are worse, it needs real strength and
perturbing weights will not get there.
"""

import json
import sys

import chess

PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material(board: chess.Board) -> int:
    total = 0
    for piece_type, value in PIECE_VALUE.items():
        total += value * len(board.pieces(piece_type, chess.WHITE))
        total -= value * len(board.pieces(piece_type, chess.BLACK))
    return total


def main() -> None:
    level = json.load(open(sys.argv[1]))["levels"][0]
    for game in level["games"]:
        board = chess.Board(game["moves"][-1]["fen"])
        us = chess.WHITE if game["color"] == "white" else chess.BLACK
        edge = material(board) * (1 if us == chess.WHITE else -1)
        print(f"=== L4 as {game['color']}: {game['outcome']} by {game['reason']} "
              f"in {game['move_count']} moves, points {game['points']}")
        print(f"  final fen : {board.fen()}")
        print(f"  material  : {edge:+d} pawns from our side")
        print(f"  last moves: {' '.join(m['uci'] for m in game['moves'][-14:])}")

        # Where did the repetition start? Walk forward and find the first position seen three times.
        replay, counts = chess.Board(game["starting_fen"]), {}
        for index, move in enumerate(game["moves"]):
            replay.push_uci(move["uci"])
            key = replay._transposition_key()
            counts[key] = counts.get(key, 0) + 1
            if counts[key] == 2:
                first_repeat = index
                break
        else:
            first_repeat = None
        if first_repeat is not None:
            at = chess.Board(game["moves"][first_repeat]["fen"])
            print(f"  shuffling from ply {first_repeat} (material {material(at) * (1 if us == chess.WHITE else -1):+d})")
        print()


if __name__ == "__main__":
    main()
