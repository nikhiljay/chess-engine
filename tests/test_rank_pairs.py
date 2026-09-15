import chess
import torch

from chess_nn.encoding import feature_indices
from chess_nn.model import rank_loss
from chess_nn.rank_pairs import MIN_GAP_CP, build_pairs, child_features, is_validation, to_arrays

START = chess.STARTING_FEN


def decision(best="e2e4", played="a2a3", best_cp=40, played_cp=40 - MIN_GAP_CP, fen=START, game=0, repeats=False):
    return {"game": game, "fen": fen, "best": best, "played": played, "best_cp": best_cp, "played_cp": played_cp, "repeats": repeats}


def test_pair_holds_the_positions_each_move_leads_to():
    [pair] = build_pairs([decision()], "diag")
    after_best = chess.Board()
    after_best.push_uci("e2e4")
    after_played = chess.Board()
    after_played.push_uci("a2a3")
    assert pair["good"] == feature_indices(after_best)
    assert pair["bad"] == feature_indices(after_played)
    assert pair["gap_cp"] == MIN_GAP_CP and pair["game"] == "diag:0"


def test_pairs_are_skipped_when_uninformative():
    assert build_pairs([decision(played="e2e4")], "d") == []  # same move
    assert build_pairs([decision(played_cp=40 - MIN_GAP_CP + 1)], "d") == []  # gap too small
    assert build_pairs([decision(best_cp=-500, played_cp=-900)], "d") == []  # already lost
    checking = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
    assert child_features(checking, "d8h4") is None  # mate: game over after the move
    assert build_pairs([decision(fen=checking, best="d8h4", played="a7a6", best_cp=10000, played_cp=0)], "d") == []


def test_arrays_pad_to_68_features():
    arrays = to_arrays(build_pairs([decision(), decision(game=1, repeats=True)], "d"))
    assert arrays["good"].shape == (2, 68) and arrays["bad"].shape == (2, 68)
    assert arrays["repeats"].tolist() == [False, True]


def test_validation_split_is_by_game_and_stable():
    keys = [f"diag:{i}" for i in range(1000)]
    val = [k for k in keys if is_validation(k)]
    assert 50 < len(val) < 150
    assert val == [k for k in keys if is_validation(k)]


def test_rank_loss_rewards_scoring_the_better_child_lower():
    good_lower = rank_loss(torch.tensor([-1.0]), torch.tensor([1.0]))
    good_higher = rank_loss(torch.tensor([1.0]), torch.tensor([-1.0]))
    assert good_lower < good_higher


def test_repetition_rule_pairs_a_shuffle_with_progress_whatever_the_gap():
    start = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"
    shuffle = {
        "game": 3, "start_fen": start, "history": ["e1e2", "e8e7", "e2e1", "e7e8"], "fen": start,
        "best": "a1a7", "best_cp": 900, "played": "e1e2", "played_cp": 880, "repeats": True,
    }
    [pair] = build_pairs([shuffle], "d")
    assert pair["kind"] == "repetition" and pair["gap_cp"] == 20
    assert build_pairs([shuffle], "d", repetition_pairs=False) == []
    assert build_pairs([{**shuffle, "best_cp": 200, "played_cp": 180}], "d") == []  # not winning
    assert build_pairs([{**shuffle, "best": "e1d1", "played": "e1e2"}], "d")[0]["kind"] == "repetition"  # d1 is new
    assert build_pairs([{**shuffle, "best": "e1e2", "played": "e1d1", "repeats": False}], "d") == []
