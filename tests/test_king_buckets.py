import chess
import torch

from chess_nn.encoding import HALF_SIZE, dense_numpy, encode_sparse
from chess_nn.export import export_onnx, parity_check, validate_with_harness
from chess_nn.harness import VALIDATE_MODEL, load_harness_tool
from chess_nn.model import EvalNet, KingBucketNet, default_king_buckets
from conftest import random_positions


def dense(boards):
    return torch.from_numpy(dense_numpy(encode_sparse(boards)))


def test_default_buckets_split_by_side_and_rank_band():
    buckets = default_king_buckets()
    assert len(buckets) == 64 and set(buckets) == set(range(8))
    assert buckets[chess.E1] == 4 and buckets[chess.C1] == 0  # back rank, kingside vs queenside
    assert buckets[chess.G2] == 5 and buckets[chess.D4] == 2 and buckets[chess.H8] == 7


def test_buckets_follow_each_sides_own_king():
    model = KingBucketNet(hidden=8)
    board = chess.Board("6k1/8/8/8/3K4/8/8/8 w - - 0 1")  # white king d4, black king g8
    x = dense([board])
    # STM (white) king d4 → queenside, ranks 3-4; NSTM (black) king g8 is g1 from its view → kingside back rank.
    assert model.buckets(x[:, :HALF_SIZE]).item() == 2
    assert model.buckets(x[:, HALF_SIZE:]).item() == 4


def test_zero_deltas_match_plain_evalnet(rng):
    torch.manual_seed(0)
    plain = EvalNet(hidden=32)
    bucketed = KingBucketNet(hidden=32)
    bucketed.load_state_dict(plain.state_dict(), strict=False)
    x = dense(random_positions(rng, 64))
    with torch.no_grad():
        assert torch.allclose(plain(x), bucketed(x), atol=1e-6)


def test_export_matches_pytorch_and_passes_harness(tmp_path, rng):
    torch.manual_seed(0)
    model = KingBucketNet(hidden=24)
    with torch.no_grad():
        model.ft.weight.mul_(20)
        model.ft_delta.normal_(0, 0.5)
        model.ft_bias_delta.normal_(0, 0.2)
    path = tmp_path / "model.onnx"
    export_onnx(model, path)
    x = dense_numpy(encode_sparse(random_positions(rng, 256)))
    assert parity_check(model, path, x) < 1e-4
    assert validate_with_harness(path)
    tool = load_harness_tool(VALIDATE_MODEL, "validate_model")
    assert tool.count_onnx_params(str(path)) == model.exported_params()


def test_deltas_get_gradients(rng):
    torch.manual_seed(0)
    model = KingBucketNet(hidden=16)
    with torch.no_grad():
        model.ft.weight.mul_(5)
    model(dense(random_positions(rng, 64))).sum().backward()
    assert model.ft_delta.grad is not None and model.ft_delta.grad.abs().sum() > 0
