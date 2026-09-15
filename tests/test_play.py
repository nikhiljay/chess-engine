import chess
import pytest
import torch

from chess_nn.export import export_onnx
from chess_nn.harness import PLAY_MOVE_BIN
from chess_nn.model import EvalNet
from chess_nn.play import game_result, parse_human_move, play, render


def test_parse_human_move_accepts_san_and_uci():
    board = chess.Board()
    assert parse_human_move(board, "e4") == chess.Move.from_uci("e2e4")
    assert parse_human_move(board, " Nf3 ") == chess.Move.from_uci("g1f3")
    assert parse_human_move(board, "d2d4") == chess.Move.from_uci("d2d4")
    assert parse_human_move(board, "e5") is None
    assert parse_human_move(board, "e2e5") is None
    assert parse_human_move(board, "hello") is None


def test_harness_castling_notation_normalizes():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert board.parse_uci("e1h1") == chess.Move.from_uci("e1g1")
    assert board.parse_uci("e1a1") == chess.Move.from_uci("e1c1")


def test_render_orients_board_for_each_side():
    white = render(chess.Board(), chess.WHITE).splitlines()
    black = render(chess.Board(), chess.BLACK).splitlines()
    assert white[0].split() == list("abcdefgh")
    assert white[1].startswith("8 ♜")
    assert black[0].split() == list("hgfedcba")
    assert black[1].startswith("1 ♖")


def test_game_result_end_conditions():
    mated = chess.Board()
    for san in ["f3", "e5", "g4", "Qh4#"]:
        mated.push_san(san)
    assert game_result(mated) == ("0-1", "checkmate")

    assert game_result(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")) == ("1/2-1/2", "stalemate")

    repeated = chess.Board()
    for san in ["Nf3", "Nf6", "Ng1", "Ng8"] * 2:
        repeated.push_san(san)
    assert game_result(repeated) == ("1/2-1/2", "threefold repetition")

    assert game_result(chess.Board()) is None


@pytest.fixture
def tiny_model(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "model.onnx"
    export_onnx(EvalNet(hidden=8), path)
    return path


def _script(*lines):
    it = iter(lines)
    return lambda _prompt: next(it)


@pytest.mark.skipif(not PLAY_MOVE_BIN.exists(), reason="harness play-move binary not built")
def test_scripted_game_against_net(tiny_model):
    output = []
    board = play(tiny_model, chess.WHITE, input_fn=_script("e4", "bogus", "quit"), out=output.append)
    assert [m.uci() for m in board.move_stack][0] == "e2e4"
    assert len(board.move_stack) == 2
    assert any(line.startswith("Net plays") for line in output)
    assert any("Not a legal move: 'bogus'" in line for line in output)


@pytest.mark.skipif(not PLAY_MOVE_BIN.exists(), reason="harness play-move binary not built")
def test_net_moves_first_as_white_and_undo(tiny_model):
    board = play(tiny_model, chess.BLACK, input_fn=_script("undo", "quit"), out=lambda _: None)
    # Net's opening move stays: undo needs a full move pair.
    assert len(board.move_stack) == 1

    board = play(tiny_model, chess.WHITE, input_fn=_script("e4", "undo", "quit"), out=lambda _: None)
    assert len(board.move_stack) == 0
