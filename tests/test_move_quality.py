import json

from chess_nn.move_quality import LOSS_CAP_CP, sample_decisions, summarize


def row(best_cp, chosen_cp):
    return {"best_cp": best_cp, "chosen_cp": chosen_cp, "agree": best_cp == chosen_cp}


def test_summary_separates_turning_moves_from_large_drops():
    rows = [
        row(50, 50),  # agreed
        row(0, -450),  # defensible → losing: turning move and large drop
        row(900, 400),  # large drop, still winning
        row(-500, -9000),  # already lost: capped loss, not a turning move
        row(-80, -120),  # small loss
    ]
    s = summarize(rows)
    assert s["decisions"] == 5 and s["agreement"] == 0.2
    assert s["defensible_decisions"] == 4 and s["turning_moves"] == 1 and s["turning_rate"] == 0.25
    assert s["large_drop_rate"] == 0.6
    assert s["mean_loss_cp"] == (0 + 450 + 500 + LOSS_CAP_CP + 40) / 5


def test_sample_is_seeded_and_bounded(tmp_path):
    lines = [json.dumps({"game": i, "ply": i}) for i in range(100)]
    (tmp_path / "decisions.jsonl").write_text("\n".join(lines) + "\n")
    first = sample_decisions(tmp_path, 10, seed=3)
    assert len(first) == 10 and first == sample_decisions(tmp_path, 10, seed=3)
    assert first != sample_decisions(tmp_path, 10, seed=4)
