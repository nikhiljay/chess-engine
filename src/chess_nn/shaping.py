"""Progress shaping for clearly won positions.

A depth-1 search picks the move whose resulting position evaluates best, so
when every move keeps a big advantage the net shuffles pieces back and forth
into threefold repetition. A small bonus for irreversible progress — advancing
the winning side's pawns, trading material off, and (once the losing side has
no pawns and little material) driving the losing king to the edge with the
winning king close — makes those moves evaluate higher than shuffles.
"""

import numpy as np

# Material per piece type in plane order P, N, B, R, Q, K.
PIECE_VALUES = np.array([1, 3, 3, 5, 9, 0], dtype=np.int32)
# Both sides' non-king material in the starting position.
FULL_MATERIAL = 78
# Mop-up applies once the losing side has no pawns and at most a rook's worth.
MOPUP_MAX_LOSING_MATERIAL = 5


def progress_bonus(
    features: np.ndarray,
    stm_winning: np.ndarray,
    pawn_cp: int = 10,
    trade_cp: int = 4,
    mopup_cp: int = 15,
) -> np.ndarray:
    """Non-negative bonus in centipawns for the winning side of each position.

    features: int16 [N, 68] active indices; only the side-to-move half is used.
    stm_winning: bool [N], whether the side to move is the winning side.
    """
    idx = features.astype(np.int32)
    valid = (idx >= 0) & (idx < 768)
    plane = np.where(valid, idx // 64, -1)
    square = np.where(valid, idx % 64, 0)
    # Side-to-move orientation: rank 0 is the side to move's back rank.
    rank, file = square // 8, square % 8
    piece = np.where(valid, plane % 6, -1)
    is_stm_piece = plane < 6
    winner_is_stm = stm_winning[:, None]
    on_winning_side = valid & (is_stm_piece == winner_is_stm)
    on_losing_side = valid & (is_stm_piece != winner_is_stm)

    # Side-to-move pawns advance up the ranks, the opponent's down.
    advancement = np.where(is_stm_piece, rank - 1, 6 - rank)
    pawn_push = np.where(on_winning_side & (piece == 0), advancement, 0).sum(axis=1)

    values = np.where(valid, PIECE_VALUES[np.clip(piece, 0, 5)], 0)
    traded = FULL_MATERIAL - values.sum(axis=1)

    losing_material = np.where(on_losing_side, values, 0).sum(axis=1)
    losing_pawns = (on_losing_side & (piece == 0)).sum(axis=1)
    mopup_active = (losing_pawns == 0) & (losing_material <= MOPUP_MAX_LOSING_MATERIAL)

    def king_square(side_mask):
        king = side_mask & (piece == 5)
        return np.where(king, rank, 0).sum(axis=1), np.where(king, file, 0).sum(axis=1)

    win_rank, win_file = king_square(on_winning_side)
    lose_rank, lose_file = king_square(on_losing_side)
    # 0 for the four center squares, 3 on the edge.
    losing_king_edge = np.maximum(np.abs(2 * lose_rank - 7), np.abs(2 * lose_file - 7)) // 2
    king_distance = np.maximum(np.abs(win_rank - lose_rank), np.abs(win_file - lose_file))
    mopup = np.where(mopup_active, losing_king_edge + (7 - king_distance), 0)

    return (pawn_cp * pawn_push + trade_cp * traded + mopup_cp * mopup).astype(np.float32)
