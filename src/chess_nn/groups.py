"""Sibling groups: the quiescence leaves one decision's legal moves are scored by.

The harness search picks a move by comparing these leaves against each other, so
error that varies between them changes which move is played, while error shared by
all of them cancels. Training can therefore penalise the varying part on its own
(chess_nn.model.group_consistency_loss) while the absolute targets stay in charge
of the level.

Groups come from `chess_nn.decision_data --groups`.
"""

import json
from pathlib import Path

import numpy as np


def position_key(fen: str) -> str:
    """Board, side to move, castling and en passant — the same key decision_data dedupes on."""
    return " ".join(fen.split()[:4])


def load_group_rows(group_files: list[str], fens: np.ndarray) -> list[list[int]]:
    """Row indices, per decision, of the leaves that are present in the loaded shards.

    Labeling drops positions whose best move is not quiet, so a group keeps only
    what survived.
    """
    row_of: dict[str, int] = {}
    for row, fen in enumerate(fens):
        key = position_key(fen.decode() if isinstance(fen, (bytes, np.bytes_)) else str(fen))
        row_of.setdefault(key, row)

    groups: list[list[int]] = []
    for path in group_files:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            keys = dict.fromkeys(position_key(fen) for fen in record["positions"])
            rows = [row_of[key] for key in keys if key in row_of]
            if len(rows) >= 2:
                groups.append(rows)
    return groups


def padded(groups: list[list[int]], keep: np.ndarray, max_members: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """[G, M] row indices and a [G, M] validity mask, over the rows where `keep` holds.

    Restricting to one split keeps a group's loss and its diagnostic on separate data.
    Groups with fewer than two surviving members carry no within-group signal.
    """
    kept = [[row for row in group if keep[row]][:max_members] for group in groups]
    kept = [group for group in kept if len(group) >= 2]
    width = max((len(group) for group in kept), default=1)
    indices = np.zeros((len(kept), width), dtype=np.int64)
    mask = np.zeros((len(kept), width), dtype=bool)
    for i, group in enumerate(kept):
        indices[i, : len(group)] = group
        mask[i, : len(group)] = True
    return indices, mask
