"""Sparse board encoding matching engine/src/nn.rs board_to_tensor().

A position activates at most 68 of the 1540 inputs (32 pieces in each of the
two perspective halves, plus 4 castling flags), so positions are stored as
active-index lists and densified in batches.
"""

import chess
import numpy as np
import torch

HALF_SIZE = 770
TENSOR_SIZE = 1540
MAX_FEATURES = 68

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def _half_indices(board: chess.Board, perspective: chess.Color, offset: int) -> list[int]:
    flip = perspective == chess.BLACK
    indices = []
    for channel, piece_type in enumerate(PIECE_ORDER):
        for color, plane in ((perspective, channel), (not perspective, channel + 6)):
            for square in chess.scan_forward(board.pieces_mask(piece_type, color)):
                if flip:
                    square = chess.square_mirror(square)
                indices.append(offset + plane * 64 + square)
    if board.has_kingside_castling_rights(perspective):
        indices.append(offset + 768)
    if board.has_queenside_castling_rights(perspective):
        indices.append(offset + 769)
    return indices


def feature_indices(board: chess.Board) -> list[int]:
    """Active input indices: STM half first, then NSTM half."""
    return _half_indices(board, board.turn, 0) + _half_indices(board, not board.turn, HALF_SIZE)


def mirror_permutation() -> np.ndarray:
    """Feature index map for mirroring the board left-right (files a<->h).

    Chess is exactly symmetric under this reflection except for castling, which is not
    file-symmetric, so only positions without castling rights may be mirrored. En passant
    is not part of the encoding, so nothing else changes and the Stockfish label carries
    over unchanged. Used as free augmentation: the nets overfit the human mix (finding 72).
    """
    permutation = np.arange(TENSOR_SIZE, dtype=np.int64)
    for half in (0, HALF_SIZE):
        for plane in range(12):
            base = half + plane * 64
            for square in range(64):
                file, rank = square % 8, square // 8
                permutation[base + square] = base + (7 - file) + 8 * rank
    return permutation


CASTLING_FEATURES = (768, 769, HALF_SIZE + 768, HALF_SIZE + 769)


def encode_sparse(boards: list[chess.Board]) -> np.ndarray:
    """Encode boards as int16 [N, 68] active indices, padded with -1."""
    features = np.full((len(boards), MAX_FEATURES), -1, dtype=np.int16)
    for row, board in enumerate(boards):
        indices = feature_indices(board)
        features[row, : len(indices)] = indices
    return features


def dense_from_sparse(features: torch.Tensor) -> torch.Tensor:
    """Densify [N, 68] index rows into float [N, 1540] on the features' device."""
    idx = features.long()
    idx = torch.where(idx < 0, TENSOR_SIZE, idx)
    dense = torch.zeros(idx.shape[0], TENSOR_SIZE + 1, device=idx.device)
    dense.scatter_(1, idx, 1.0)
    return dense[:, :TENSOR_SIZE]


def dense_numpy(features: np.ndarray) -> np.ndarray:
    """Densify [N, 68] index rows into a contiguous float32 [N, 1540] array."""
    idx = np.where(features < 0, TENSOR_SIZE, features).astype(np.int64)
    dense = np.zeros((features.shape[0], TENSOR_SIZE + 1), dtype=np.float32)
    np.put_along_axis(dense, idx, 1.0, axis=1)
    return np.ascontiguousarray(dense[:, :TENSOR_SIZE])
