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
from chess_nn.model import KING_PLANE, EvalNet, KingBucketNet

# name → (fen, predicate on the eval in pawns)
SANITY_POSITIONS = {
    "startpos": ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", lambda v: abs(v) < 0.5),
    "queen_up": ("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", lambda v: v > 5.0),
    "queen_down": ("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", lambda v: v < -5.0),
}


def _weight(name: str, tensor: torch.Tensor) -> onnx.TensorProto:
    return numpy_helper.from_array(tensor.detach().cpu().numpy().astype(np.float32), name)


def _int64(name: str, values: list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.array(values, dtype=np.int64), name)


def _transformer(model: EvalNet, half: str, out: str) -> tuple[list[onnx.TensorProto], list[onnx.NodeProto]]:
    """Initializers (shared by both halves, so only emitted for 'stm') and nodes for one half."""
    if not isinstance(model, KingBucketNet):
        initializers = [_weight("ft.weight", model.ft.weight), _weight("ft.bias", model.ft.bias)] if half == "stm" else []
        nodes = [
            helper.make_node("Gemm", [half, "ft.weight", "ft.bias"], [f"{out}_lin"], transB=1),
            helper.make_node("Clip", [f"{out}_lin", "clip_min", "clip_max"], [out]),
        ]
        return initializers, nodes

    initializers = []
    if half == "stm":
        weight, bias = model.merged_transformer()
        initializers = [
            _weight("ft.bucket_weight", weight),
            _weight("ft.bucket_bias", bias),
            _int64("bucket_of_square", model.bucket_of_square.tolist()),
            _int64("king_start", [KING_PLANE]),
            _int64("king_end", [KING_PLANE + 64]),
            _int64("king_axis", [1]),
            _int64("axis1", [1]),
        ]
    # One bucket's weights per position via Gather, so inference costs one
    # transformer multiply regardless of the bucket count.
    nodes = [
        helper.make_node("Slice", [half, "king_start", "king_end", "king_axis"], [f"{out}_king"]),
        helper.make_node("ArgMax", [f"{out}_king"], [f"{out}_square"], axis=1, keepdims=0),
        helper.make_node("Gather", ["bucket_of_square", f"{out}_square"], [f"{out}_bucket"], axis=0),
        helper.make_node("Gather", ["ft.bucket_weight", f"{out}_bucket"], [f"{out}_w"], axis=0),
        helper.make_node("Gather", ["ft.bucket_bias", f"{out}_bucket"], [f"{out}_b"], axis=0),
        helper.make_node("Unsqueeze", [half, "axis1"], [f"{out}_row"]),
        helper.make_node("MatMul", [f"{out}_row", f"{out}_w"], [f"{out}_mm"]),
        helper.make_node("Squeeze", [f"{out}_mm", "axis1"], [f"{out}_flat"]),
        helper.make_node("Add", [f"{out}_flat", f"{out}_b"], [f"{out}_lin"]),
        helper.make_node("Clip", [f"{out}_lin", "clip_min", "clip_max"], [out]),
    ]
    return initializers, nodes


def export_onnx(model: EvalNet, path) -> None:
    initializers = [
        numpy_helper.from_array(np.array([HALF_SIZE, HALF_SIZE], dtype=np.int64), "split_sizes"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), "clip_min"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), "clip_max"),
    ]
    nodes = [helper.make_node("Split", ["board", "split_sizes"], ["stm", "nstm"], axis=1)]
    for half, out in (("stm", "act_stm"), ("nstm", "act_nstm")):
        half_inits, half_nodes = _transformer(model, half, out)
        initializers += half_inits
        nodes += half_nodes
    nodes.append(helper.make_node("Concat", ["act_stm", "act_nstm"], ["accumulator"], axis=1))
    x = "accumulator"
    for i, layer in enumerate(model.head):
        initializers += [_weight(f"head{i}.weight", layer.weight), _weight(f"head{i}.bias", layer.bias)]
        nodes += [
            helper.make_node("Gemm", [x, f"head{i}.weight", f"head{i}.bias"], [f"head{i}_out"], transB=1),
            helper.make_node("Relu", [f"head{i}_out"], [f"head{i}_act"]),
        ]
        x = f"head{i}_act"
    initializers += [_weight("out.weight", model.out.weight), _weight("out.bias", model.out.bias)]
    nodes.append(helper.make_node("Gemm", [x, "out.weight", "out.bias"], ["eval"], transB=1))

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
