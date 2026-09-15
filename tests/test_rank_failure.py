import chess

from chess_nn.rank_failure import searched_move, summarize
from chess_nn.search_check import move_value

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def material(board: chess.Board) -> float:
    return float(sum(v * (len(board.pieces(t, board.turn)) - len(board.pieces(t, not board.turn))) for t, v in VALUES.items()))


def test_origin_separates_static_children_from_capture_lines_and_checks():
    board = chess.Board("4k3/8/8/4p3/8/8/3Q4/4K3 w - - 0 1")
    quiet = searched_move(material, board, chess.Move.from_uci("e1e2"))
    hanging = searched_move(material, board, chess.Move.from_uci("d2d4"))
    check = searched_move(material, board, chess.Move.from_uci("d2d7"))
    assert quiet["origin"] == "static child"
    assert hanging["origin"] == "capture line" and hanging["value"] == -1.0
    assert check["origin"] == "check"
    assert board.fen() == "4k3/8/8/4p3/8/8/3Q4/4K3 w - - 0 1"  # the board is restored


def test_searched_value_matches_the_harness_move_value():
    board = chess.Board("r3k2r/8/8/3q4/4P3/8/8/R3K2R w KQkq - 0 1")
    for move in list(board.legal_moves)[:12]:
        assert searched_move(material, board, move)["value"] == move_value(material, board, move)


def test_summary_attributes_each_turning_move_to_the_larger_shift():
    rows = [
        {"origin": "static child", "refutation_kind": "quiet", "losing_shift": 0.8, "safe_shift": -0.1, "replica_matches_play_move": True},
        {"origin": "capture line", "refutation_kind": "capture list", "losing_shift": 0.1, "safe_shift": -0.6, "replica_matches_play_move": True},
    ]
    summary = summarize(rows, matched=100, prevented=1)
    assert summary["cause"] == {"overrated losing move": 1, "underrated safe move": 1}
    assert summary["origin_by_refutation"] == {"static child / quiet": 1, "capture line / capture list": 1}
    assert summary["turning_introduced"] == 2 and summary["turning_prevented"] == 1
