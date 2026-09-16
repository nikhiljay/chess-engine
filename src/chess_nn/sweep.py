"""Train and evaluate a list of runs from a JSON spec, skipping finished steps.

Spec:
  {
    "shards": ["data/shards/shard_0000.npz"],         # paths or globs under training/
    "levels": [1, 2],
    "openings": "data/dev_openings_100.txt",        # optional; defaults to the dev set
    "defaults": {"epochs": 20, "batch_size": 4096},   # TrainConfig fields
    "runs": [{"name": "h128-s800", "hidden": 128, "wdl_scale_cp": 800}]
  }

Usage:
  uv run python -m chess_nn.sweep sweeps/shard0.json
"""

import argparse
import json
from pathlib import Path

from chess_nn.evaluate import evaluate_model, tag_for
from chess_nn.harness import DEV_OPENINGS, TRAINING_ROOT
from chess_nn.report import collect_runs, format_table
from chess_nn.train import TrainConfig, run_training


def expand_shards(patterns: list[str]) -> list[str]:
    paths: list[Path] = []
    for pattern in patterns:
        if any(c in pattern for c in "*?["):
            paths += sorted(TRAINING_ROOT.glob(pattern))
        else:
            paths.append(TRAINING_ROOT / pattern)
    shards = [str(p) for p in paths if ".tmp" not in p.name]
    if not shards:
        raise SystemExit(f"no shards match {patterns}")
    return shards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    shards = expand_shards(spec["shards"])
    openings = TRAINING_ROOT / spec["openings"] if "openings" in spec else DEV_OPENINGS
    levels = spec.get("levels", [1])
    tag = tag_for(openings, official=False)

    for run in spec["runs"]:
        # A run may name its own shards, so one spec can compare data or label sets.
        run_shards = expand_shards(run["shards"]) if "shards" in run else shards
        cfg = TrainConfig(**{**spec.get("defaults", {}), **run, "shards": run_shards})
        run_dir = Path(cfg.runs_dir) / cfg.name
        if not (run_dir / "summary.json").exists():
            run_training(cfg)
        missing = [level for level in levels if not (run_dir / f"{tag}_L{level}.json").exists()]
        if missing:
            evaluate_model(run_dir / "model.onnx", missing, openings, tag)
        print(format_table(collect_runs()), flush=True)


if __name__ == "__main__":
    main()
