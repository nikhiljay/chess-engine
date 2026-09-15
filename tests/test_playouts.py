import chess

from chess_nn.playouts import distance_value, pick_decisions, playout, playout_rows

MATE_IN_ONE = "k7/8/1K6/8/8/8/8/7Q w - - 0 1"  # Qh8#


def scripted(moves: list[str], cp: int = 900):
    replies = iter(moves)
    return lambda position: (cp, next(replies))


def test_distance_value_keeps_far_positions_and_rises_toward_mate():
    assert distance_value(900, plies_to_mate=150, slope_cp=15, horizon_plies=100) == 900
    assert distance_value(900, plies_to_mate=90, slope_cp=15, horizon_plies=100) == 1050
    assert distance_value(900, plies_to_mate=1, slope_cp=15, horizon_plies=100) == 2385
    assert distance_value(-200, plies_to_mate=150, slope_cp=15, horizon_plies=100) == 0
    assert distance_value(2400, plies_to_mate=1, slope_cp=15, horizon_plies=100) == 2500


def test_playout_ends_in_mate_and_labels_both_sides_by_distance():
    game = playout(scripted(["h1h7", "a8b8", "h7b7"]), "k7/8/1K6/8/8/8/8/7Q w - - 0 1", [], max_plies=10)
    assert game["result"] == "mate" and game["winner"] == chess.WHITE and len(game["steps"]) == 3
    rows = playout_rows(game, slope_cp=15, horizon_plies=100)
    values = [score for _, score, _, _ in rows]
    # White to move 3 plies before mate, Black to move 2 plies before, White to move 1 ply before.
    assert values == [900 + 15 * 97, -(0 + 15 * 98), 900 + 15 * 99]
    assert values[2] > values[0] > 0 > values[1]


def test_captures_are_dropped_and_draws_or_unfinished_games_give_nothing():
    capture = playout(scripted(["h1a8"]), "q7/8/1K6/8/8/8/8/k6Q w - - 0 1", [], max_plies=1)
    assert capture["result"] == "unfinished" and playout_rows(capture, 15, 100) == []
    game = {"result": "mate", "winner": chess.WHITE, "steps": [(MATE_IN_ONE, 900, True, chess.WHITE)]}
    assert playout_rows(game, 15, 100) == []


def test_history_is_replayed_before_the_playout(tmp_path):
    start = chess.STARTING_FEN
    game = playout(scripted(["f8c5", "d1h5", "g8f6", "h5f7"]), start, ["e2e4", "e7e5", "f1c4"], max_plies=10)
    assert game["result"] == "mate" and game["winner"] == chess.WHITE
    assert game["steps"][0][0] == chess.Board("rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 2").fen()


def test_pick_decisions_samples_winning_decisions_per_game(tmp_path):
    rows = [{"game": g, "ply": p, "best_cp": cp, "fen": "x"} for g in (0, 1) for p, cp in ((1, 100), (3, 600), (5, 700), (7, 800))]
    (tmp_path / "decisions.jsonl").write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\n")
    picked = pick_decisions([tmp_path], min_cp=500, per_game=2, seed=0)
    assert len(picked) == 4 and all(d["best_cp"] >= 500 for d in picked)
    assert sorted({d["game"] for d in picked}) == [0, 1]
