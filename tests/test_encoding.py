import re
import subprocess

import chess
import numpy as np
import pytest
import torch

from chess_nn.encoding import TENSOR_SIZE, dense_from_sparse, dense_numpy, encode_sparse, feature_indices
from chess_nn.harness import DUMP_ENCODING_BIN, REFERENCE_ENCODING, load_harness_tool
from conftest import random_positions

EDGE_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
    "8/P7/8/8/8/8/7p/K6k b - - 0 1",
]


@pytest.fixture(scope="module")
def reference():
    return load_harness_tool(REFERENCE_ENCODING, "reference_encoding")


def _dense(board: chess.Board) -> np.ndarray:
    return dense_numpy(encode_sparse([board]))[0]


def test_edge_positions_match_reference(reference):
    for fen in EDGE_FENS:
        board = chess.Board(fen)
        np.testing.assert_array_equal(_dense(board), reference.board_to_tensor(board), err_msg=fen)


def test_random_positions_match_reference(reference, rng):
    for board in random_positions(rng, 300):
        np.testing.assert_array_equal(_dense(board), reference.board_to_tensor(board), err_msg=board.fen())


def _rust_indices(fen: str) -> set[int]:
    output = subprocess.run([str(DUMP_ENCODING_BIN), fen], capture_output=True, text=True, check=True).stdout
    return {int(i) for i in re.findall(r"^\s+\[(\d+)\] = 1$", output, flags=re.MULTILINE)}


@pytest.mark.skipif(not DUMP_ENCODING_BIN.exists(), reason="harness dump-encoding binary not built")
def test_positions_match_rust_encoder(rng):
    fens = EDGE_FENS + [board.fen() for board in random_positions(rng, 100)]
    for fen in fens:
        assert set(feature_indices(chess.Board(fen))) == _rust_indices(fen), fen


def test_torch_and_numpy_densify_agree(rng):
    features = encode_sparse(random_positions(rng, 50))
    via_torch = dense_from_sparse(torch.from_numpy(features)).numpy()
    np.testing.assert_array_equal(via_torch, dense_numpy(features))
    assert via_torch.shape == (50, TENSOR_SIZE)
