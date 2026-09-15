"""Run the harness compete binary on a model and summarize the results.

Iteration uses dev openings. --official runs the harness's own 25 openings and
is for confirming milestones only. Requested levels play in parallel.

Usage:
  uv run python -m chess_nn.evaluate runs/v1-h256/model.onnx --levels 1 2
  uv run python -m chess_nn.evaluate runs/v1-h256/model.onnx --levels 1 --official
"""

import argparse
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess

from chess_nn.export import OnnxEvaluator
from chess_nn.harness import COMPETE_BIN, DEV_OPENINGS, HARNESS_ROOT, OFFICIAL_OPENINGS

# A repetition draw counts as a thrown-away win when the net's own eval of the
# final position is above this many pawns for the net's side.
REPETITION_WINNING_PAWNS = 2.0


def tag_for(openings: Path, official: bool) -> str:
    """Result file prefix: 'official', or the openings stem minus '_openings' (dev_openings_100 → dev_100)."""
    if official:
        return "official"
    return openings.stem.replace("_openings", "")


def run_compete(model: Path, level: int, openings: Path, json_path: Path) -> dict:
    if not COMPETE_BIN.exists():
        raise FileNotFoundError(f"{COMPETE_BIN} missing; run `cargo build --release -p cli` in {HARNESS_ROOT}")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.unlink(missing_ok=True)
    cmd = [
        str(COMPETE_BIN),
        str(model.resolve()),
        "--level", str(level),
        "--openings", str(openings.resolve()),
        "--json-output", str(json_path.resolve()),
    ]
    # compete exits 1 when the level fails, which is a result, not an error.
    with open(json_path.with_suffix(".log"), "w") as log:
        subprocess.run(cmd, cwd=HARNESS_ROOT, check=False, stdout=log, stderr=subprocess.STDOUT)
    return json.loads(json_path.read_text())


def summarize_level(level: dict, evaluator: OnnxEvaluator | None) -> dict:
    games = level["games"]
    # compete's own W/D/L tally counts draws as losses (0.5 as u32 == 0), so
    # recount from each game's points.
    points = [game["points"] for game in games]
    thrown_wins = 0
    for game in games:
        if game["reason"] != "repetition":
            continue
        fen = game["moves"][-1]["fen"] if game["moves"] else game["starting_fen"]
        board = chess.Board(fen)
        value = float(evaluator.evaluate([board])[0])
        nn_color = chess.WHITE if game["color"] == "white" else chess.BLACK
        if board.turn != nn_color:
            value = -value
        if value > REPETITION_WINNING_PAWNS:
            thrown_wins += 1
    return {
        "level": level["level"],
        "score": level["score"],
        "score_pct": level["score_pct"],
        "wins": sum(p == 1.0 for p in points),
        "draws": sum(p == 0.5 for p in points),
        "losses": sum(p == 0.0 for p in points),
        "passed": level["passed"],
        "end_reasons": dict(Counter(game["reason"] for game in games)),
        "repetition_while_winning": thrown_wins,
    }


def evaluate_model(
    model: Path, levels: list[int], openings: Path, tag: str, out_dir: Path | None = None
) -> list[dict]:
    """Play the levels in parallel and merge their summaries into {tag}_summary.json."""
    out_dir = out_dir or model.parent
    evaluator = OnnxEvaluator(model)
    with ThreadPoolExecutor(max_workers=len(levels)) as pool:
        results = list(pool.map(
            lambda level: run_compete(model, level, openings, out_dir / f"{tag}_L{level}.json"), levels
        ))

    summary_path = out_dir / f"{tag}_summary.json"
    existing = json.loads(summary_path.read_text()) if summary_path.exists() else []
    merged = {entry["level"]: entry for entry in existing}
    for result in results:
        entry = summarize_level(result["levels"][0], evaluator)
        merged[entry["level"]] = entry
    summaries = [merged[level] for level in sorted(merged)]
    summary_path.write_text(json.dumps(summaries, indent=2))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("--levels", type=int, nargs="+", default=[1])
    parser.add_argument("--official", action="store_true", help="use the official openings (milestones only)")
    parser.add_argument("--openings", type=Path, help="override the openings file")
    parser.add_argument("--out-dir", type=Path, help="defaults to the model's directory")
    args = parser.parse_args()

    openings = args.openings or (OFFICIAL_OPENINGS if args.official else DEV_OPENINGS)
    tag = tag_for(openings, args.official)
    summaries = evaluate_model(args.model, args.levels, openings, tag, args.out_dir)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
