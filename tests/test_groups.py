"""Sibling-group indexing and the within-group consistency loss."""

import json

import numpy as np
import torch

from chess_nn.groups import load_group_rows, padded
from chess_nn.model import group_consistency_loss

A = "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1"
B = "8/8/4k3/8/8/4K3/4P3/8 b - - 0 1"
C = "8/8/5k2/8/8/4K3/4P3/8 w - - 0 1"
D = "8/8/3k4/8/8/4K3/4P3/8 w - - 0 1"


def fens(*values: str) -> np.ndarray:
    return np.asarray([v.encode() for v in values])


def test_shared_error_is_free_but_varying_error_is_not():
    target = torch.zeros(1, 3)
    mask = torch.ones(1, 3, dtype=torch.bool)
    shifted = torch.full((1, 3), 4.0)  # every sibling wrong by the same amount
    assert group_consistency_loss(shifted, target, mask).item() == 0.0
    varying = torch.tensor([[1.0, -1.0, 0.0]])
    assert group_consistency_loss(varying, target, mask).item() > 0.5


def test_targets_are_centipawns_like_the_other_losses():
    # Predictions in pawns against targets in centipawns: all three are exactly right.
    pred = torch.tensor([[1.0, -2.0, 3.0]])
    target_cp = torch.tensor([[100.0, -200.0, 300.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    assert group_consistency_loss(pred, target_cp, mask).item() == 0.0


def test_padding_is_ignored():
    target = torch.zeros(1, 3)
    both = torch.tensor([[1.0, -1.0, 99.0]])
    mask = torch.tensor([[True, True, False]])
    # Only the two valid members count: errors +1 and -1, variance 1.
    assert group_consistency_loss(both, target, mask).item() == 1.0


def test_load_group_rows_maps_positions_to_shard_rows(tmp_path):
    path = tmp_path / "groups.jsonl"
    path.write_text(
        json.dumps({"decision": A, "positions": [B, C]}) + "\n"
        + json.dumps({"decision": B, "positions": [C, "8/8/8/8/8/4K3/4P3/4k3 w - - 0 1"]}) + "\n"
    )
    groups = load_group_rows([str(path)], fens(A, B, C))
    # The first decision's leaves are both present; the second keeps only one, so it is dropped.
    assert groups == [[1, 2]]


def test_padded_splits_by_row_membership():
    groups = [[0, 1, 2], [2, 3]]
    keep = np.array([True, True, False, False])
    indices, mask = padded(groups, keep)
    assert indices.shape == (1, 2) and mask.sum() == 2
    assert sorted(indices[0].tolist()) == [0, 1]
    # A group with one surviving member carries no within-group signal.
    assert len(padded(groups, np.array([True, False, False, True]))[0]) == 0
