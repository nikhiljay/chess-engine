import chess

from chess_nn.decision_data import decision_positions
from chess_nn.quiescence_audit import INF, Shifted, audit_decision, decision_errors, harness_evaluated, move_meta, summarize
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
        chosen, evaluated = harness_evaluated(material, chess.Board(fen))
        assert chosen == search_choice(material, chess.Board(fen))
        assert evaluated


def test_shifting_the_decisive_leaf_changes_the_choice_and_a_forced_move_never_changes():
    # White's only legal move is Kh2, so no evaluation error can change it.
    forced = chess.Board("6r1/8/8/8/8/8/8/k6K w - - 0 1")
    chosen, evaluated = harness_evaluated(material, forced)
    assert chosen.uci() == "h1h2"
    assert set(decision_errors(material, forced, chosen, evaluated).values()) == {INF}
    # Rook e1 takes the undefended pawn on e5 without giving check (+5 against +4 for quiet moves),
    # so the quiet child it leads to is kept, evaluated at the root, and decisive at a small error.
    board = chess.Board("7k/8/8/4p3/8/8/8/K3R3 w - - 0 1")
    chosen, evaluated = harness_evaluated(material, board)
    assert chosen.uci() == "e1e5"
    after = board.copy()
    after.push(chosen)
    errors = decision_errors(material, board, chosen, evaluated)
    assert errors[after.fen()] <= 4.0  # the kept leaf of the chosen move is decisive at a 4-pawn error


def test_shifted_net_moves_only_the_named_position():
    board = chess.Board(FENS[0])
    other = chess.Board(FENS[1])
    shifted = Shifted(material, board.fen(), 2.5)
    assert shifted(board) == material(board) + 2.5 and shifted(other) == material(other)


def test_audit_splits_evaluated_positions_into_kept_and_dropped():
    fen = FENS[0]
    audit = audit_decision(material, fen)
    assert audit["kept_positions"] == len(set(decision_positions(material, fen)[0]))
    assert set(audit["roles"]) <= {"kept", "dropped move child", "dropped deeper"}
    assert sum(len(v) for v in audit["roles"].values()) == audit["evaluated_by_harness"]
    report = summarize([audit])
    assert report["decisions"] == 1 and report["kept_positions"] == audit["kept_positions"]


def test_move_meta_marks_the_played_move_and_how_its_value_was_reached():
    decision = {"fen": FENS[0], "played": "d2d4", "best": "e1e2"}
    rows = {r["move"]: r for r in move_meta(material, decision)}
    assert rows["d2d4"]["played"] and rows["d2d4"]["kind"] == "leaf" and rows["d2d4"]["value"] == -1.0
    assert rows["e1e2"]["best"] and rows["e1e2"]["kind"] == "child"
