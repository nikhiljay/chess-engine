import chess
import numpy as np

from chess_nn.encoding import encode_sparse
from chess_nn.shaping import progress_bonus
from chess_nn.train import training_targets


def bonus(fen: str, stm_winning: bool = True, **weights) -> float:
    features = encode_sparse([chess.Board(fen)])
    return float(progress_bonus(features, np.array([stm_winning]), **weights)[0])


def test_no_bonus_in_starting_position():
    assert bonus(chess.STARTING_FEN) == 0


def test_advanced_pawns_score_higher_for_winning_side():
    assert bonus("4k3/8/4P3/8/8/8/8/4K3 w - - 0 1") > bonus("4k3/8/8/8/8/4P3/8/4K3 w - - 0 1")
    # Black to move and winning: a black pawn on e3 is further advanced than on e6.
    assert bonus("4k3/8/8/8/8/4p3/8/4K3 b - - 0 1") > bonus("4k3/8/4p3/8/8/8/8/4K3 b - - 0 1")
    # White to move but Black winning: Black's advanced pawn still counts.
    assert bonus("4k3/8/8/8/8/4p3/8/4K3 w - - 0 1", stm_winning=False) > bonus(
        "4k3/8/4p3/8/8/8/8/4K3 w - - 0 1", stm_winning=False
    )


def test_losing_side_pawns_earn_nothing():
    assert bonus("4k3/8/8/8/8/4p3/8/4K3 w - - 0 1", trade_cp=0, mopup_cp=0) == 0


def test_trades_increase_bonus():
    no_knights = "r1bqkb1r/pppppppp/8/8/8/8/PPPPPPPP/R1BQKB1R w KQkq - 0 1"
    assert bonus(no_knights, pawn_cp=0, mopup_cp=0) == 4 * 12


def test_mopup_rewards_cornered_king_and_close_winning_king():
    center = "8/8/8/3k4/8/8/8/KQ6 w - - 0 1"
    cornered = "k7/8/1K6/8/8/8/8/1Q6 w - - 0 1"
    assert bonus(cornered, pawn_cp=0, trade_cp=0) > bonus(center, pawn_cp=0, trade_cp=0)
    # Inactive while the losing side still has pawns.
    assert bonus("k7/p7/1K6/8/8/8/8/1Q6 w - - 0 1", pawn_cp=0, trade_cp=0) == 0


def test_training_targets_apply_shaping_only_to_won_non_mate_positions():
    boards = [
        chess.Board("4k3/8/4P3/8/8/8/8/4K3 w - - 0 1"),  # white winning, advanced pawn
        chess.Board("4k3/8/4P3/8/8/8/8/4K3 b - - 0 1"),  # same, black to move and losing
        chess.Board("4k3/8/4P3/8/8/8/8/4K3 w - - 0 1"),  # below threshold
        chess.Board("4k3/8/4P3/8/8/8/8/4K3 w - - 0 1"),  # mate score
    ]
    features = encode_sparse(boards)
    scores = np.array([800, -800, 100, 32000], dtype=np.int16)
    mates = np.array([0, 0, 0, 5], dtype=np.int16)
    plain = training_targets(scores, mates, 2500)
    shaped = training_targets(scores, mates, 2500, features=features, shaping_threshold_cp=300)
    assert shaped[0] > plain[0]
    assert shaped[1] < plain[1]
    assert shaped[2] == plain[2]
    assert shaped[3] == plain[3]
