import numpy as np
import onnx
import torch

from chess_nn.encoding import dense_numpy, encode_sparse
from chess_nn.export import export_onnx, parity_check, sanity_checks, validate_with_harness
from chess_nn.harness import VALIDATE_MODEL, load_harness_tool
from chess_nn.model import EvalNet, count_params
from conftest import random_positions


def test_param_counts_match_spec():
    assert count_params(EvalNet(256)) == 214_881
    assert count_params(EvalNet(128)) == 108_001


def test_export_passes_harness_validation(tmp_path):
    torch.manual_seed(0)
    model = EvalNet(hidden=16)
    path = tmp_path / "model.onnx"
    export_onnx(model, path)

    assert validate_with_harness(path)
    loaded = onnx.load(str(path))
    assert loaded.ir_version == 8
    assert loaded.opset_import[0].version == 17
    # Only the Split sizes initializer adds counted parameters beyond the weights.
    tool = load_harness_tool(VALIDATE_MODEL, "validate_model")
    assert tool.count_onnx_params(str(path)) == count_params(model) + 2


def test_onnx_matches_pytorch(tmp_path, rng):
    torch.manual_seed(0)
    model = EvalNet(hidden=64)
    # Push the feature transformer into its active range so clipping is exercised.
    with torch.no_grad():
        model.ft.weight.mul_(20)
    path = tmp_path / "model.onnx"
    export_onnx(model, path)
    dense = dense_numpy(encode_sparse(random_positions(rng, 256)))
    assert parity_check(model, path, dense) < 1e-4


def test_sanity_checks_report_all_positions(tmp_path):
    model = EvalNet(hidden=8)
    path = tmp_path / "model.onnx"
    export_onnx(model, path)
    result = sanity_checks(path)
    assert set(result) == {"startpos", "queen_up", "queen_down"}
    assert all(np.isfinite(entry["eval"]) for entry in result.values())
