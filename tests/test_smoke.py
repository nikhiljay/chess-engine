"""End-to-end: label → shard → train → export → one harness level-1 game pair."""

import random
import shutil

import pytest

from chess_nn.evaluate import run_compete
from chess_nn.harness import COMPETE_BIN
from chess_nn.label import Stockfish, label_position
from chess_nn.shards import write_shard
from chess_nn.train import TrainConfig, run_training
from conftest import random_positions

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("stockfish") is None, reason="stockfish not installed"),
    pytest.mark.skipif(not COMPETE_BIN.exists(), reason="harness compete binary not built"),
]

ITALIAN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"


def test_end_to_end(tmp_path):
    engine = Stockfish()
    try:
        rows = [label_position(engine, b.fen(), depth=1) for b in random_positions(random.Random(7), 1000)]
    finally:
        engine.close()
    rows = [row for row in rows if row is not None]
    assert len(rows) > 300

    shard = tmp_path / "shard_0000.npz"
    write_shard(shard, rows)
    cfg = TrainConfig(
        name="smoke",
        shards=[str(shard)],
        hidden=32,
        epochs=1,
        batch_size=256,
        runs_dir=str(tmp_path / "runs"),
    )
    run_dir = run_training(cfg)
    assert (run_dir / "model.onnx").exists()

    openings = tmp_path / "openings.txt"
    openings.write_text(ITALIAN + "\n")
    result = run_compete(run_dir / "model.onnx", 1, openings, tmp_path / "compete.json")
    assert len(result["levels"][0]["games"]) == 2
