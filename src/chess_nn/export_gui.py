"""Export a trained run for the play page in training/gui.

Writes gui/data/weights.json — the network's shapes and its float32 weights as
base64, laid out for the TypeScript engine (feature-transformer rows per input
feature) — and gui/data/reference.json: positions with the harness's play-move
choice and ONNX eval, which `npm run check` compares the engine against.

Usage:
  uv run python -m chess_nn.export_gui runs/jj6-h1024-s1000-cp20
  (cd gui && npm run check && npm run build)
"""

import argparse
import base64
import json
import random
from pathlib import Path

import chess
import numpy as np
import torch

from chess_nn.export import OnnxEvaluator
from chess_nn.harness import DEV_OPENINGS, SHARDS_DIR, TRAINING_ROOT, read_opening_fens
from chess_nn.model import EvalNet, count_params
from chess_nn.play import net_move

GUI_DATA = TRAINING_ROOT / "gui" / "data"

# Rule cases the engine must match: back-rank mate, promotion, en passant,
# castling rights on both sides, and a king in check.
SPECIAL_FENS = [
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    "8/P7/8/8/8/8/6k1/4K3 w - - 0 1",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "r3k2r/pppq1ppp/2n2n2/3pp3/1b1PP3/2N2N2/PPPQ1PPP/R3KB1R w KQkq - 0 8",
    "4k3/8/8/8/8/8/4q3/4K3 w - - 0 1",
]


def load_run(run_dir: Path) -> EvalNet:
    config = json.loads((run_dir / "config.json").read_text())
    if config.get("king_buckets"):
        raise SystemExit("the play page engine does not support king-bucket models yet")
    model = EvalNet(config["hidden"], tuple(config["head"]))
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    return model.eval()


def weight_payload(model: EvalNet) -> tuple[list[dict], bytes]:
    """Tensor manifest and concatenated little-endian float32 weights."""
    state = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    tensors = [("ft_by_feature", np.ascontiguousarray(state["ft.weight"].T)), ("ft_bias", state["ft.bias"])]
    for i in range(len(model.head)):
        tensors += [(f"head{i}_w", state[f"head.{i}.weight"]), (f"head{i}_b", state[f"head.{i}.bias"])]
    tensors += [("out_w", state["out.weight"]), ("out_b", state["out.bias"])]
    manifest, blob, offset = [], bytearray(), 0
    for name, array in tensors:
        manifest.append({"name": name, "shape": list(array.shape), "offset": offset, "length": int(array.size)})
        blob += array.astype("<f4").tobytes()
        offset += int(array.size)
    return manifest, bytes(blob)


def reference_fens(seed: int = 7) -> list[str]:
    """Dev openings, sampled shard positions across piece counts, and SPECIAL_FENS."""
    rng = random.Random(seed)
    fens = read_opening_fens(DEV_OPENINGS)[:15]
    with np.load(SHARDS_DIR / "shard_0000.npz") as shard:
        shard_fens = shard["fens"]
        pool = [f.decode() for f in shard_fens[rng.sample(range(len(shard_fens)), 400)]]
    by_pieces = sorted(pool, key=lambda f: chess.Board(f).occupied.bit_count())
    return fens + by_pieces[:15] + by_pieces[190:200] + by_pieces[-10:] + SPECIAL_FENS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="run directory with config.json, model.pt and model.onnx")
    parser.add_argument("--out", type=Path, default=GUI_DATA)
    args = parser.parse_args()

    run_dir = args.run.resolve()
    model = load_run(run_dir)
    manifest, blob = weight_payload(model)
    summary_path = run_dir / "dev_100_summary.json"
    levels = json.loads(summary_path.read_text()) if summary_path.exists() else []
    meta = {
        "run": run_dir.name,
        "params": count_params(model),
        "hidden": model.hidden,
        "head": [layer.out_features for layer in model.head],
        "dev": {f"L{level['level']}": level["score_pct"] for level in levels},
        "tensors": manifest,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "weights.json").write_text(json.dumps({"meta": meta, "weights_b64": base64.b64encode(blob).decode()}))

    onnx_path = run_dir / "model.onnx"
    boards = [chess.Board(fen) for fen in reference_fens()]
    evals = OnnxEvaluator(onnx_path).evaluate(boards)
    reference = [
        {"fen": board.fen(), "harness_uci": net_move(onnx_path, board).uci(), "onnx_eval": round(float(value), 5)}
        for board, value in zip(boards, evals)
        if not board.is_game_over()
    ]
    (args.out / "reference.json").write_text(json.dumps(reference, indent=1))
    print(f"{run_dir.name}: {meta['params']:,} params → {args.out}/weights.json; {len(reference)} reference positions")


if __name__ == "__main__":
    main()
