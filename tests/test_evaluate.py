from pathlib import Path

import torch

from chess_nn.evaluate import summarize_level, tag_for
from chess_nn.model import wdl_loss
from chess_nn.train import parse_head


def test_summary_recounts_draws_from_points():
    level = {
        "level": 1, "score": 1.5, "score_pct": 50.0, "passed": False,
        "wins": 1, "draws": 0, "losses": 2,  # compete's buggy tally
        "games": [
            {"points": 1.0, "reason": "checkmate"},
            {"points": 0.5, "reason": "max_plies"},
            {"points": 0.0, "reason": "checkmate"},
        ],
    }
    summary = summarize_level(level, evaluator=None)
    assert (summary["wins"], summary["draws"], summary["losses"]) == (1, 1, 1)
    assert summary["end_reasons"] == {"checkmate": 2, "max_plies": 1}


def test_tag_for_openings_files():
    assert tag_for(Path("data/dev_openings.txt"), official=False) == "dev"
    assert tag_for(Path("data/dev_openings_100.txt"), official=False) == "dev_100"
    assert tag_for(Path("data/openings.txt"), official=True) == "official"


def test_parse_head():
    assert parse_head("32,32") == [32, 32]
    assert parse_head("none") == []
    assert parse_head("") == []


def test_larger_wdl_scale_separates_big_advantages():
    evals = torch.tensor([8.0])
    target = torch.tensor([2500.0])
    assert wdl_loss(evals, target, scale_cp=800) > wdl_loss(evals, target, scale_cp=400)
