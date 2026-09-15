import chess
import torch

from chess_nn.encoding import HALF_SIZE, dense_numpy, encode_sparse
from chess_nn.export import export_onnx, parity_check, validate_with_harness
from chess_nn.model import ThreatNet
from chess_nn.threats import THREAT_FEATURES, attack_maps, geometry, threat_features
from conftest import random_positions


def geo():
    return {name: torch.from_numpy(matrix) for name, matrix in geometry().items()}


def perspective_square(square: int, perspective: chess.Color) -> int:
    file, rank = chess.square_file(square), chess.square_rank(square)
    return file + (rank if perspective == chess.WHITE else 7 - rank) * 8


def test_attack_counts_match_python_chess(rng):
    boards = random_positions(rng, 40)
    dense = torch.from_numpy(dense_numpy(encode_sparse(boards)))
    g = geo()
    for half_index, perspective_of in ((0, lambda b: b.turn), (1, lambda b: not b.turn)):
        half = dense[:, half_index * HALF_SIZE : (half_index + 1) * HALF_SIZE]
        maps = attack_maps(half, g)
        for i, board in enumerate(boards):
            me = perspective_of(board)
            for square in chess.SQUARES:
                idx = perspective_square(square, me)
                assert maps["own_attacks"][i, idx].item() == len(board.attackers(me, square)), (board.fen(), square)
                assert maps["their_attacks"][i, idx].item() == len(board.attackers(not me, square)), (board.fen(), square)


def test_hanging_piece_is_flagged():
    # White to move; the black knight on e5 is attacked by the d4 pawn and undefended.
    board = chess.Board("4k3/8/8/4n3/3P4/8/8/4K3 w - - 0 1")
    features = threat_features(torch.from_numpy(dense_numpy(encode_sparse([board])))[:, :HALF_SIZE], geo())
    assert features.shape == (1, THREAT_FEATURES)
    per_plane = features.reshape(1, 36, 64)
    e5 = perspective_square(chess.E5, chess.WHITE)
    knight = 1
    their_attacked = 6 + 6 + 5  # plane offset of "their pieces I attack"
    their_undefended = their_attacked + 6
    their_by_pawn = their_undefended + 6
    assert per_plane[0, their_attacked + knight, e5] == 1
    assert per_plane[0, their_undefended + knight, e5] == 1
    assert per_plane[0, their_by_pawn + knight - 1, e5] == 1


def test_threat_net_exports_with_parity_and_passes_harness(tmp_path, rng):
    torch.manual_seed(0)
    model = ThreatNet(hidden=16, head=(8,), threat_hidden=8)
    with torch.no_grad():
        model.ft.weight.mul_(20)
        model.threat_ft.weight.mul_(20)
    path = tmp_path / "model.onnx"
    export_onnx(model, path)
    dense = dense_numpy(encode_sparse(random_positions(rng, 128)))
    assert parity_check(model, path, dense) < 1e-4
    assert validate_with_harness(path)
