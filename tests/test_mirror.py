"""The left-right mirror map must match python-chess's own flip."""

import chess
import numpy as np

from chess_nn.encoding import CASTLING_FEATURES, feature_indices, mirror_permutation

# Positions without castling rights, where the reflection is exact.
POSITIONS = [
    "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
    "r4rk1/pp3ppp/2n5/3q4/3P4/2N2N2/PP3PPP/R4RK1 w - - 0 15",
    "8/5pk1/6p1/7p/7P/6P1/5PK1/3r4 b - - 0 40",
    "2b5/8/1k6/8/8/5N2/6K1/8 w - - 0 60",
]


def test_mirror_matches_python_chess_flip():
    permutation = mirror_permutation()
    for fen in POSITIONS:
        board = chess.Board(fen)
        mirrored = board.transform(chess.flip_horizontal)
        mapped = sorted(int(permutation[i]) for i in feature_indices(board))
        assert mapped == sorted(feature_indices(mirrored)), fen


def test_mirror_is_an_involution():
    permutation = mirror_permutation()
    assert np.array_equal(permutation[permutation], np.arange(len(permutation)))


def test_castling_features_are_left_alone():
    permutation = mirror_permutation()
    for feature in CASTLING_FEATURES:
        assert permutation[feature] == feature


def test_castling_positions_are_detected():
    # The caller must skip these: a mirrored king cannot keep its castling rights.
    start = feature_indices(chess.Board())
    assert any(feature in start for feature in CASTLING_FEATURES)
    assert not any(feature in feature_indices(chess.Board(POSITIONS[0])) for feature in CASTLING_FEATURES)
