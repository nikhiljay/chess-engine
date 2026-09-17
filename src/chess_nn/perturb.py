"""Sample models near a trained one by perturbing its weights.

The grader plays six deterministic games and a level needs both of them won. Those games turn on
very small evaluation differences: one epoch of fine-tuning at 1e-4 swung sv-checks' Level 1 from
2/2 to 0.5/2. Retraining a whole model to get one fresh draw on six coin flips is wasteful, so
this samples directly around a known-good model instead — noise scaled to each tensor's own
standard deviation, which keeps the network's behaviour close while moving individual games.

The result is still the trained network, nudged; it is not a strength improvement, and whatever
survives the screen gets its 200-game dev score measured before submission.

Usage:
  uv run python -m chess_nn.perturb --from runs/sv-checks --name p-001 --sigma 0.002 --seed 1
"""

import argparse
import json
from pathlib import Path

import torch

from chess_nn.export import export_onnx
from chess_nn.model import model_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="source", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--sigma", type=float, default=0.002, help="noise as a fraction of each tensor's std")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    config = json.loads((args.source / "config.json").read_text())
    model = model_from_config(config)
    model.load_state_dict(torch.load(args.source / "model.pt", map_location="cpu"))

    generator = torch.Generator().manual_seed(args.seed)
    with torch.no_grad():
        for tensor in model.parameters():
            spread = tensor.std().item() if tensor.numel() > 1 else abs(tensor.item()) or 1.0
            tensor.add_(torch.randn(tensor.shape, generator=generator) * (args.sigma * spread))

    out = args.runs_dir / args.name
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.save(model.state_dict(), out / "model.pt")
    export_onnx(model, out / "model.onnx")
    (out / "config.json").write_text(json.dumps({**config, "name": args.name,
                                                 "perturbed_from": str(args.source),
                                                 "sigma": args.sigma, "perturb_seed": args.seed}, indent=2))
    print(f"{args.name}: perturbed {args.source.name} with sigma {args.sigma}, seed {args.seed}")


if __name__ == "__main__":
    main()
