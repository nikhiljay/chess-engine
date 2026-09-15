import shutil

import chess
import pytest

from chess_nn.label import SCORE_LIMIT_CP, Stockfish, is_quiet_best_move, label_position, parse_score


def test_parse_score_keeps_raw_centipawns():
    assert parse_score("info depth 9 seldepth 12 multipv 1 score cp 34 nodes 5000 pv e2e4") == (34, 0)
    assert parse_score("info depth 9 score cp -9000 nodes 1") == (-9000, 0)
    assert parse_score("info depth 9 score cp 40000 nodes 1") == (SCORE_LIMIT_CP, 0)


def test_parse_score_keeps_mate_distance():
    assert parse_score("info depth 9 score mate 3 nodes 1") == (SCORE_LIMIT_CP, 3)
    assert parse_score("info depth 9 score mate -2 nodes 1") == (-SCORE_LIMIT_CP, -2)


def test_parse_score_ignores_bounds_and_strings():
    assert parse_score("info depth 9 score cp 50 lowerbound nodes 1") is None
    assert parse_score("info depth 9 score cp 50 upperbound nodes 1") is None
    assert parse_score("info string NNUE evaluation using nn-big.nnue") is None


def test_quiet_move_filter():
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
    assert not is_quiet_best_move(board, "e4d5")  # capture
    assert is_quiet_best_move(board, "g1f3")
    ep = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    assert not is_quiet_best_move(ep, "e5f6")  # en passant
    promo = chess.Board("8/P7/8/8/8/8/7p/K6k w - - 0 1")
    assert not is_quiet_best_move(promo, "a7a8q")


class FakeEngine:
    """Returns scripted (score, best move) per depth and records requested depths."""

    def __init__(self, by_depth):
        self.by_depth = by_depth
        self.depths = []

    def analyse(self, fen, depth):
        self.depths.append(depth)
        return self.by_depth[depth]


def test_deep_search_replaces_lopsided_scores():
    engine = FakeEngine({9: ((900, 0), "g1f3"), 14: ((32000, 5), "b1c3")})
    fen, score, mate, _ = label_position(engine, chess.STARTING_FEN, 9, deep_depth=14, deep_threshold_cp=600)
    assert (score, mate) == (32000, 5)
    assert engine.depths == [9, 14]


def test_deep_search_skips_balanced_and_mate_scores():
    balanced = FakeEngine({9: ((120, 0), "g1f3"), 14: ((999, 0), "g1f3")})
    assert label_position(balanced, chess.STARTING_FEN, 9, deep_depth=14)[1] == 120
    assert balanced.depths == [9]
    mate = FakeEngine({9: ((32000, 3), "g1f3"), 14: ((32000, 2), "g1f3")})
    assert label_position(mate, chess.STARTING_FEN, 9, deep_depth=14)[2] == 3
    assert mate.depths == [9]


def test_deep_search_best_move_decides_quiet_filter():
    # Deeper search finds a capture as best move, so the position is dropped.
    engine = FakeEngine({9: ((900, 0), "g1f3"), 14: ((1200, 0), "e2e4")})
    board_fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
    engine.by_depth[14] = ((1200, 0), "e4d5")
    assert label_position(engine, board_fen, 9, deep_depth=14) is None


@pytest.mark.skipif(shutil.which("stockfish") is None, reason="stockfish not installed")
def test_stockfish_labels_startpos():
    engine = Stockfish()
    try:
        row = label_position(engine, chess.STARTING_FEN, depth=6)
        assert row is not None
        fen, score, mate, indices = row
        assert fen == chess.STARTING_FEN
        assert abs(score) < 100
        assert mate == 0
        assert len(indices) == 68
        # A hanging queen: best move captures it, so the position is dropped.
        assert label_position(engine, "rnb1kbnr/pppp1ppp/8/4p3/3qP3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1", depth=6) is None
    finally:
        engine.close()
