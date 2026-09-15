from chess_nn.diagnose import BLUNDER_CP, MATE_CP, report, summarize_game, to_cp


def decision(ply, best_cp, played_cp, best="e2e4", played="e2e4", repeats=False):
    return {"ply": ply, "best": best, "best_cp": best_cp, "played": played, "played_cp": played_cp, "repeats": repeats}


def game(result="1/2-1/2", end="repetition", net_white=True, plies=60):
    return {"result": result, "end": end, "net_white": net_white, "level": 2, "moves": ["e2e4"] * plies}


def test_to_cp_orders_mates_beyond_centipawns():
    assert to_cp((120, 0)) == 120
    assert to_cp((32000, 3)) == MATE_CP - 3
    assert to_cp((-32000, -2)) == -MATE_CP + 2
    assert to_cp((32000, 3)) > to_cp((32000, 5)) > to_cp((2400, 0))
    assert to_cp(None) == 0


def test_slip_is_first_drawish_move_while_winning():
    decisions = [
        decision(10, 50, 40),
        decision(20, 450, 430, played="g1f3"),
        decision(30, 520, 0, best="a7a8q", played="h1g1", repeats=True),
        decision(40, 600, 0, repeats=True),
    ]
    summary = summarize_game(game(), decisions)
    assert summary["had_win"] and summary["first_win_ply"] == 20
    assert summary["slip"]["ply"] == 30 and summary["slip"]["repeats"]
    assert summary["outcome"] == "draw"


def test_blunder_needs_a_big_drop_from_a_position_not_already_lost():
    lost = game(result="0-1", end="checkmate")
    summary = summarize_game(lost, [decision(12, 30, 20), decision(18, 80, 80 - BLUNDER_CP - 5), decision(24, -900, -9000)])
    assert summary["outcome"] == "loss" and summary["blunder"]["ply"] == 18
    assert summarize_game(lost, [decision(12, 30, -100)])["blunder"] is None


def test_report_counts_repetition_wins_and_slips():
    summaries = [
        summarize_game(game(), [decision(20, 400, 380), decision(30, 400, 0, repeats=True)]),
        summarize_game(game(), [decision(20, 50, 40)]),
        summarize_game(game(result="1-0", end="checkmate"), [decision(20, 700, 700)]),
    ]
    result = report(summaries)
    assert result["repetition"] == {
        "games": 2,
        "had_win": 1,
        "slipped": 1,
        "slip_move_repeats_position": 1,
        "median_plies_slip_to_end": 30,
        "median_best_eval_cp": 400,
    }
    assert result["outcomes"] == {"draw": 2, "win": 1}
