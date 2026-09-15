from chess_nn.turning import loss_breakdown


def d(ply, best_cp, played_cp):
    return {"ply": ply, "best_cp": best_cp, "played_cp": played_cp, "fen": "", "played": ""}


def test_turning_moves_are_separated_from_large_drops():
    decisions = [
        d(10, 900, 400),  # large drop, still winning
        d(20, 30, -450),  # defensible → losing: turning
        d(30, -800, -9000),  # large drop, already lost
        d(40, -60, -350),  # another turning move
    ]
    b = loss_breakdown(decisions)
    assert b["turning_moves"] == 2 and b["first_turning"]["ply"] == 20
    assert b["large_drops"] == 3 and b["large_drops_not_turning"] == 2
    assert b["ever_defensible"]


def test_lost_from_the_start_has_no_turning_move():
    b = loss_breakdown([d(10, -500, -600), d(20, -700, -1200)])
    assert b["turning_moves"] == 0 and not b["ever_defensible"] and b["large_drops"] == 1
