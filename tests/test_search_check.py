import chess

from chess_nn.search_check import MATE, harness_captures, move_value, quiescence, summarize

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def material(board: chess.Board) -> float:
    """Material balance from the side to move's view, as a stand-in evaluator."""
    score = 0
    for piece_type, value in VALUES.items():
        score += value * (len(board.pieces(piece_type, board.turn)) - len(board.pieces(piece_type, not board.turn)))
    return float(score)


def test_capture_list_matches_the_harness():
    board = chess.Board("r3k2r/8/8/3q4/4P3/8/8/R3K2R w KQkq - 0 1")
    captures = harness_captures(board)
    assert captures[0] == chess.Move.from_uci("e4d5")  # queen victim first
    assert chess.Move.from_uci("e1g1") in captures  # castling counts (queenside is illegal: the queen covers d1)
    ep = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    assert chess.Move.from_uci("e5f6") not in harness_captures(ep)


def test_quiescence_resolves_captures_but_not_quiet_mates():
    board = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    assert move_value(material, board, chess.Move.from_uci("e1e2")) < move_value(material, board, chess.Move.from_uci("e4d5"))
    # 32...Nxe4 wins a rook statically; the search never looks at the quiet 33. Rh8#.
    diag = chess.Board("r2b1rk1/pb3q2/1p3n2/2pNB1p1/2P1R1B1/P3PP2/1PQ5/2K4R b - - 0 32")
    assert move_value(material, diag, chess.Move.from_uci("f6e4")) > -MATE / 2
    mated = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 b - - 0 1")
    mated.push_uci("g8h8")
    assert quiescence(material, mated, -float("inf"), float("inf")) > -MATE / 2  # Rd8# is a quiet move


def test_summary_separates_static_search_and_choice():
    rows = [
        {"kind": "score", "decisive": True, "static": True, "search": False, "chosen": "worse"},
        {"kind": "score", "decisive": False, "static": True, "search": True, "chosen": "better"},
        {"kind": "mate_in_one", "decisive": False, "static": False, "search": False, "chosen": "other"},
    ]
    s = summarize(rows)
    assert s["all"]["pairs"] == 3 and s["score"]["static_right_search_wrong"] == 1
    assert s["decisive"]["chose_worse"] == 1.0 and s["mate_in_one"]["chose_other"] == 1.0
