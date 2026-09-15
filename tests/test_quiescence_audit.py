import chess

from chess_nn.decision_data import decision_positions
from chess_nn.quiescence_audit import INF, audit_decision, harness_evaluated, move_meta, summarize, thresholds
from chess_nn.search_check import search_choice

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
FENS = [
    "4k3/8/8/4p3/8/8/3Q4/4K3 w - - 0 1",
    "r3k2r/8/8/3q4/4P3/8/8/R3K2R w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
]


def material(board: chess.Board) -> float:
    return float(sum(v * (len(board.pieces(t, board.turn)) - len(board.pieces(t, not board.turn))) for t, v in VALUES.items()))


def test_harness_replica_picks_the_search_check_move():
    for fen in FENS:
        board = chess.Board(fen)
        chosen, evaluated = harness_evaluated(material, board)
        assert chosen == search_choice(material, chess.Board(fen))
        assert evaluated  # every root child not in check is evaluated


def test_thresholds_follow_gaps_and_unique_best_lines():
    # A root child (opponent to move) stands pat at 1 but captures into a line worth 3; the chosen move leads
    # the runner-up by 2, so the child's value has to rise by more than 2 to change the decision.
    leaf = {"fen": "leaf", "ply": 2, "stand_pat": -3.0, "children": [], "value": -3.0}
    child = {"fen": "child", "ply": 1, "stand_pat": 1.0, "children": [leaf], "value": 3.0}
    out = []
    thresholds(child, 2.0, INF, out)
    assert out == [("child", 1, 4.0), ("leaf", 2, 2.0)]


def test_audit_splits_positions_into_kept_and_dropped():
    fen = FENS[0]
    audit = audit_decision(material, fen)
    kept = set(decision_positions(material, fen)[0])
    assert audit["kept_positions"] == len(kept)
    assert set(audit["roles"]) <= {"kept", "dropped move child", "dropped deeper"}
    assert len(audit["roles"]["kept"]) >= len(kept)  # a kept position can recur in several trees
    report = summarize([audit], skipped=0)
    assert report["roles"]["dropped move child"]["positions"] >= 1  # Qd4's child, searched past to exd4


def test_move_meta_marks_the_played_move_and_how_its_value_was_reached():
    decision = {"fen": FENS[0], "played": "d2d4", "best": "e1e2"}
    rows = {r["move"]: r for r in move_meta(material, decision)}
    assert rows["d2d4"]["played"] and rows["d2d4"]["kind"] == "leaf" and rows["d2d4"]["value"] == -1.0
    assert rows["e1e2"]["best"] and rows["e1e2"]["kind"] == "child"
