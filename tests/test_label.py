import shutil

import chess
import pytest

from chess_nn.label import SCORE_CLAMP_CP, Stockfish, is_quiet_best_move, label_position, parse_score


def test_parse_score_centipawns_and_clamp():
    assert parse_score("info depth 9 seldepth 12 multipv 1 score cp 34 nodes 5000 pv e2e4") == 34
    assert parse_score("info depth 9 score cp -9000 nodes 1") == -SCORE_CLAMP_CP


def test_parse_score_mates_map_to_clamp():
    assert parse_score("info depth 9 score mate 3 nodes 1") == SCORE_CLAMP_CP
    assert parse_score("info depth 9 score mate -2 nodes 1") == -SCORE_CLAMP_CP


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


@pytest.mark.skipif(shutil.which("stockfish") is None, reason="stockfish not installed")
def test_stockfish_labels_startpos():
    engine = Stockfish()
    try:
        row = label_position(engine, chess.STARTING_FEN, depth=6)
        assert row is not None
        fen, score, indices = row
        assert fen == chess.STARTING_FEN
        assert abs(score) < 100
        assert len(indices) == 68
        # A hanging queen: best move captures it, so the position is dropped.
        assert label_position(engine, "rnb1kbnr/pppp1ppp/8/4p3/3qP3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1", depth=6) is None
    finally:
        engine.close()
