from pathlib import Path

import numpy as np
import torch

from chess_nn.evaluate import summarize_level, tag_for
from chess_nn.model import wdl_loss
from chess_nn.train import parse_head, training_targets


def test_training_targets_clamp_and_mate_ladder():
    scores = np.array([100, 9000, -9000, 32000, -32000, 32000], dtype=np.int16)
    mates = np.array([0, 0, 0, 3, -2, 80], dtype=np.int16)
    clamped = training_targets(scores, mates, score_clamp_cp=2500)
    assert clamped.tolist() == [100, 2500, -2500, 2500, -2500, 2500]
    laddered = training_targets(scores, mates, score_clamp_cp=2500, mate_base_cp=3000, mate_step_cp=20)
    assert laddered.tolist() == [100, 2500, -2500, 3000 + 47 * 20, -(3000 + 48 * 20), 3000]


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


def test_cp_loss_separates_big_advantages_and_clips():
    from chess_nn.model import cp_loss

    evals = torch.tensor([8.5])
    assert cp_loss(evals, torch.tensor([850.0])) == 0
    assert cp_loss(evals, torch.tensor([3980.0])) > cp_loss(evals, torch.tensor([3000.0])) > 0
    assert cp_loss(torch.tensor([50.0]), torch.tensor([32000.0])) == 0


def test_larger_wdl_scale_separates_big_advantages():
    evals = torch.tensor([8.0])
    target = torch.tensor([2500.0])
    assert wdl_loss(evals, target, scale_cp=800) > wdl_loss(evals, target, scale_cp=400)
