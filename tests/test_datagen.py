import subprocess

import chess
import pytest
import torch

from chess_nn.export import export_onnx
from chess_nn.harness import DATAGEN_BIN
from chess_nn.model import EvalNet


@pytest.mark.skipif(not DATAGEN_BIN.exists(), reason="datagen binary not built")
def test_datagen_writes_unique_legal_quiet_positions(tmp_path):
    torch.manual_seed(0)
    model = tmp_path / "model.onnx"
    export_onnx(EvalNet(hidden=8), model)
    out = tmp_path / "positions.txt"
    subprocess.run(
        [str(DATAGEN_BIN), str(model), "--out", str(out), "--games", "4", "--threads", "2",
         "--sample-prob", "1.0", "--min-ply", "10", "--random-move-prob", "0.2"],
        check=True, capture_output=True,
    )
    lines = out.read_text().splitlines()
    assert lines, "expected some positions"
    assert len(set(lines)) == len(lines)
    for line in lines:
        board = chess.Board(line)
        assert board.is_valid()
        assert not board.is_check()
        assert not board.is_game_over()


@pytest.mark.skipif(not DATAGEN_BIN.exists(), reason="datagen binary not built")
def test_datagen_loop_mode_emits_quiet_legal_children(tmp_path):
    torch.manual_seed(0)
    model = tmp_path / "model.onnx"
    export_onnx(EvalNet(hidden=8), model)
    out = tmp_path / "positions.txt"
    result = subprocess.run(
        [str(DATAGEN_BIN), str(model), "--out", str(out), "--games", "16", "--threads", "2",
         "--sample-prob", "0", "--random-move-prob", "0", "--loop-plies", "12", "--min-eval", "-100000"],
        check=True, capture_output=True, text=True,
    )
    assert "loops)" in result.stderr
    lines = out.read_text().splitlines()
    assert len(set(lines)) == len(lines)
    for line in lines:
        board = chess.Board(line)
        assert board.is_valid()
        assert not board.is_check()
        assert not board.is_game_over()
