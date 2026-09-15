"""Labeled position shards: active feature indices, Stockfish scores, FENs."""

from pathlib import Path

import numpy as np

from chess_nn.encoding import MAX_FEATURES

# (fen, score_cp from side to move, signed moves-to-mate or 0, active feature indices)
Row = tuple[str, int, int, list[int]]


def write_shard(path: Path, rows: list[Row]) -> None:
    features = np.full((len(rows), MAX_FEATURES), -1, dtype=np.int16)
    scores = np.zeros(len(rows), dtype=np.int16)
    mates = np.zeros(len(rows), dtype=np.int16)
    fens = []
    for i, (fen, score, mate, indices) in enumerate(rows):
        features[i, : len(indices)] = indices
        scores[i] = score
        mates[i] = mate
        fens.append(fen.encode())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, features=features, score_cp=scores, mate=mates, fens=np.asarray(fens))
    tmp.rename(path)


def load_shards(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate features [N, 68], score_cp [N] and mate [N] (all int16) across shards.

    Shards written before mate was stored (already clamped to ±2500) load with mate = 0.
    """
    features, scores, mates = [], [], []
    for path in paths:
        with np.load(path) as shard:
            features.append(shard["features"])
            scores.append(shard["score_cp"])
            if "mate" in shard.files:
                mates.append(shard["mate"])
            else:
                mates.append(np.zeros(len(shard["score_cp"]), dtype=np.int16))
    return np.concatenate(features), np.concatenate(scores), np.concatenate(mates)
