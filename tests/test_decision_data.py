import chess

from chess_nn.decision_data import decision_positions, quiescence_leaf
from chess_nn.search_check import move_value

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def material(board: chess.Board) -> float:
    return float(sum(v * (len(board.pieces(t, board.turn)) - len(board.pieces(t, not board.turn))) for t, v in VALUES.items()))


def test_leaf_follows_the_capture_the_search_prefers():
    # Black to move can take the undefended white queen on d4; the search ends in the quiet position after exd4.
    board = chess.Board("4k3/8/8/4p3/3Q4/8/8/4K3 b - - 0 1")
    value, leaf = quiescence_leaf(material, board, -float("inf"), float("inf"))
    after = board.copy()
    after.push_uci("e5d4")
    assert value == 1.0 and leaf == after.fen()


def test_leaf_is_the_position_itself_when_standing_pat():
    board = chess.Board("4k3/8/8/8/8/8/8/4K2R b - - 0 1")
    value, leaf = quiescence_leaf(material, board, -float("inf"), float("inf"))
    assert value == -5.0 and leaf == board.fen()


def test_leaf_value_matches_the_search_check_move_value():
    board = chess.Board("r3k2r/8/8/3q4/4P3/8/8/R3K2R w KQkq - 0 1")
    for move in list(board.legal_moves)[:12]:
        board.push(move)
        value, _ = quiescence_leaf(material, board, -float("inf"), float("inf"))
        board.pop()
        assert -value == move_value(material, board, move)


def test_decision_positions_are_the_leaves_the_search_scores_each_move_by():
    board = chess.Board("4k3/8/8/4p3/8/8/3Q4/4K3 w - - 0 1")
    positions, kinds = decision_positions(material, board.fen())
    assert board.fen() not in positions  # the decision position is not scored by this decision
    after_qd4 = board.copy()
    after_qd4.push_uci("d2d4")
    assert after_qd4.fen() not in positions  # quiescence searches past the blunder's child
    after_exd4 = after_qd4.copy()
    after_exd4.push_uci("e5d4")
    assert after_exd4.fen() in positions  # the quiet leaf the search evaluates after Qd4
    after_ke2 = board.copy()
    after_ke2.push_uci("e1e2")
    assert after_ke2.fen() in positions  # a quiet child quiescence stands pat on
    assert kinds["leaf"] >= 1 and kinds["child"] >= 1
    assert kinds["child"] + kinds["leaf"] + kinds["terminal"] == board.legal_moves.count()
