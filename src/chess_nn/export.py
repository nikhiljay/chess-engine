"""ONNX export, harness validation, parity and sanity checks.

The graph is built directly with onnx.helper so the output always matches the
harness requirements (opset 17, IR 8, named batch dims, Gemm layers) regardless
of torch exporter defaults.
"""

import chess
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper

from chess_nn.encoding import HALF_SIZE, TENSOR_SIZE, dense_numpy, encode_sparse
from chess_nn.harness import VALIDATE_MODEL, load_harness_tool
from chess_nn.model import EvalNet

# name → (fen, predicate on the eval in pawns)
SANITY_POSITIONS = {
    "startpos": ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", lambda v: abs(v) < 0.5),
    "queen_up": ("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", lambda v: v > 5.0),
    "queen_down": ("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", lambda v: v < -5.0),
}


def _weight(name: str, tensor: torch.Tensor) -> onnx.TensorProto:
    return numpy_helper.from_array(tensor.detach().cpu().numpy().astype(np.float32), name)


def export_onnx(model: EvalNet, path) -> None:
    initializers = [
        _weight("ft.weight", model.ft.weight),
        _weight("ft.bias", model.ft.bias),
        _weight("l1.weight", model.l1.weight),
        _weight("l1.bias", model.l1.bias),
        _weight("l2.weight", model.l2.weight),
        _weight("l2.bias", model.l2.bias),
        _weight("out.weight", model.out.weight),
        _weight("out.bias", model.out.bias),
        numpy_helper.from_array(np.array([HALF_SIZE, HALF_SIZE], dtype=np.int64), "split_sizes"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), "clip_min"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), "clip_max"),
    ]
    nodes = [
        helper.make_node("Split", ["board", "split_sizes"], ["stm", "nstm"], axis=1),
        helper.make_node("Gemm", ["stm", "ft.weight", "ft.bias"], ["ft_stm"], transB=1),
        helper.make_node("Gemm", ["nstm", "ft.weight", "ft.bias"], ["ft_nstm"], transB=1),
        helper.make_node("Clip", ["ft_stm", "clip_min", "clip_max"], ["act_stm"]),
        helper.make_node("Clip", ["ft_nstm", "clip_min", "clip_max"], ["act_nstm"]),
        helper.make_node("Concat", ["act_stm", "act_nstm"], ["accumulator"], axis=1),
        helper.make_node("Gemm", ["accumulator", "l1.weight", "l1.bias"], ["l1_out"], transB=1),
        helper.make_node("Relu", ["l1_out"], ["l1_act"]),
        helper.make_node("Gemm", ["l1_act", "l2.weight", "l2.bias"], ["l2_out"], transB=1),
        helper.make_node("Relu", ["l2_out"], ["l2_act"]),
        helper.make_node("Gemm", ["l2_act", "out.weight", "out.bias"], ["eval"], transB=1),
    ]
    graph = helper.make_graph(
        nodes,
        "chess_nn_eval",
        [helper.make_tensor_value_info("board", TensorProto.FLOAT, ["batch", TENSOR_SIZE])],
        [helper.make_tensor_value_info("eval", TensorProto.FLOAT, ["batch", 1])],
        initializer=initializers,
    )
    onnx_model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="chess-nn"
    )
    onnx_model.ir_version = 8
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(path))


class OnnxEvaluator:
    """Evaluate python-chess boards with an exported model via ONNX Runtime."""

    def __init__(self, path) -> None:
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def evaluate(self, boards: list[chess.Board]) -> np.ndarray:
        dense = dense_numpy(encode_sparse(boards))
        return self.session.run(None, {"board": dense})[0][:, 0]


def validate_with_harness(path) -> bool:
    tool = load_harness_tool(VALIDATE_MODEL, "validate_model")
    return tool.validate_onnx(str(path))


def parity_check(model: EvalNet, path, dense: np.ndarray) -> float:
    """Max absolute difference between PyTorch and ONNX Runtime outputs."""
    with torch.no_grad():
        expected = model.cpu().eval()(torch.from_numpy(dense)).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"board": dense})[0]
    return float(np.max(np.abs(expected - actual)))


def sanity_checks(path) -> dict[str, dict]:
    evaluator = OnnxEvaluator(path)
    names = list(SANITY_POSITIONS)
    evals = evaluator.evaluate([chess.Board(SANITY_POSITIONS[n][0]) for n in names])
    return {
        name: {"eval": round(float(value), 3), "pass": bool(SANITY_POSITIONS[name][1](value))}
        for name, value in zip(names, evals)
    }
