import pytest

from chess_nn.paired import compare_decisions, interval, opening_differences, outcomes, sign_test_p


def game(fen, color, points, reason="checkmate"):
    return {"starting_fen": fen, "color": color, "points": points, "reason": reason}


def test_opening_differences_pair_games_by_opening_and_colour():
    control = [game("a", "white", 1.0), game("a", "black", 0.0), game("b", "white", 0.5), game("b", "black", 0.5)]
    candidate = [game("a", "white", 1.0), game("a", "black", 0.5), game("b", "white", 0.0), game("b", "black", 0.5)]
    assert opening_differences(candidate, control) == [0.5, -0.5]
    with pytest.raises(ValueError):
        opening_differences(candidate[::-1], control)


def test_interval_is_per_game_with_opening_units():
    r = interval([1.0, 0.0, 1.0, 0.0], games_per_opening=2.0)
    assert r["mean_points_per_game"] == 0.25 and r["openings_better"] == 2 and r["openings_worse"] == 0
    assert r["ci95"][0] < 0.25 < r["ci95"][1]


def test_outcomes_count_draws_from_points():
    o = outcomes([game("a", "white", 1.0), game("a", "black", 0.5, "repetition"), game("b", "white", 0.0)])
    assert o["wdl"] == "1/1/1" and o["repetition_draws"] == 1 and o["score_pct"] == 50.0


def test_sign_test_is_exact_and_symmetric():
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(5, 5) == 1.0
    assert sign_test_p(10, 0) == pytest.approx(2 / 1024)
    assert sign_test_p(2, 9) == sign_test_p(9, 2)


def test_decisions_count_prevented_and_introduced_turning_moves():
    def row(game, ply, best_cp, chosen_cp, chosen="e2e4"):
        return {"game": game, "ply": ply, "best_cp": best_cp, "chosen_cp": chosen_cp, "chosen": chosen}

    control = [row(0, 1, 0, -500, "a"), row(0, 2, 50, 40), row(1, 3, -50, -350, "b"), row(2, 4, 100, 90)]
    candidate = [row(0, 1, 0, -20, "c"), row(0, 2, 50, -400, "d"), row(1, 3, -50, -350, "b"), row(9, 9, 0, 0)]
    r = compare_decisions(candidate, control)
    assert r["decisions"] == 3
    assert r["turning_prevented"] == 1 and r["turning_introduced"] == 1
    assert r["turning_candidate"] == 2 and r["turning_control"] == 2
    assert r["choices"] == {"different move": 2, "same move": 1}
