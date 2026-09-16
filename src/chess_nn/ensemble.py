"""Average several trained nets into one ONNX model.

Finding 71 measured the defect as over-confidence: in level positions the nets answer with
spread 3.26 against a truth of 1.42 and correlate only 0.35 with d9, while Stockfish's static
eval reaches 0.561. Much of that spread is independent error, and averaging independently
trained nets cancels it without any retraining. The harness caps parameters at 10M, so three
h1024 nets (2.57M) fit comfortably; inference cost scales with the number of members.

Usage:
  uv run python -m chess_nn.ensemble --runs runs/jj6-h1024-s1000-cp20 runs/rank-b-h1024-w0-seed1 \\
      --name ens-h1024-2
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from chess_nn.export import export_onnx_torch, sanity_checks, validate_with_harness
from chess_nn.harness import RUNS_DIR, VALIDATE_MODEL, load_harness_tool
from chess_nn.model import model_from_config


class Ensemble(nn.Module):
    """Mean of several evaluators, which is itself an evaluator in pawns."""

    def __init__(self, members: list[nn.Module]) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        return torch.stack([member(board) for member in self.members]).mean(dim=0)


def load_run(run: Path) -> nn.Module:
    model = model_from_config(json.loads((run / "config.json").read_text()))
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    return model.eval()


def build(runs: list[Path], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ensemble = Ensemble([load_run(run) for run in runs]).eval()
    onnx_path = out_dir / "model.onnx"
    export_onnx_torch(ensemble, onnx_path)

    first = json.loads((runs[0] / "config.json").read_text())
    (out_dir / "config.json").write_text(json.dumps({**first, "name": out_dir.name, "members": [str(r) for r in runs]}, indent=2))
    torch.save(ensemble.state_dict(), out_dir / "model.pt")
    summary = {
        "params": load_harness_tool(VALIDATE_MODEL, "validate_model").count_onnx_params(str(onnx_path)),
        "members": [run.name for run in runs],
        "harness_validation": validate_with_harness(onnx_path),
        "sanity": sanity_checks(onnx_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--name", required=True, help="run directory to create under runs/")
    args = parser.parse_args()
    print(json.dumps(build(args.runs, Path(RUNS_DIR) / args.name), indent=2))


if __name__ == "__main__":
    main()
