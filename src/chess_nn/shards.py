"""Labeled position shards: active feature indices, Stockfish scores, FENs."""

from pathlib import Path

import numpy as np

from chess_nn.encoding import MAX_FEATURES

# (fen, score_cp from side to move, active feature indices)
Row = tuple[str, int, list[int]]


def write_shard(path: Path, rows: list[Row]) -> None:
    features = np.full((len(rows), MAX_FEATURES), -1, dtype=np.int16)
    scores = np.zeros(len(rows), dtype=np.int16)
    fens = []
    for i, (fen, score, indices) in enumerate(rows):
        features[i, : len(indices)] = indices
        scores[i] = score
        fens.append(fen.encode())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, features=features, score_cp=scores, fens=np.asarray(fens))
    tmp.rename(path)


def load_shards(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate features [N, 68] int16 and score_cp [N] int16 across shards."""
    features, scores = [], []
    for path in paths:
        with np.load(path) as shard:
            features.append(shard["features"])
            scores.append(shard["score_cp"])
    return np.concatenate(features), np.concatenate(scores)
