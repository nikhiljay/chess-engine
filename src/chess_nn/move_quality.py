"""Move quality through the actual search on a representative sample of decisions.

Pair benchmarks hold only positions where some model erred, so they cannot tell
whether a model is worse at ordinary moves. This takes a seeded random sample
of every decision in held-out games (chess_nn.diagnose output, which carries
the game history and Stockfish's best move and score), has each run choose a
move with the harness's play-move binary, and scores that choice with
Stockfish at the same depth, with the game history. Per run:

  - agreement with Stockfish's best move,
  - mean centipawn loss (each loss capped at LOSS_CAP_CP),
  - large drops (loss >= BLUNDER_CP),
  - turning moves: from a defensible position (best >= DEFENSIBLE_CP) to a
    losing one (chosen <= -WIN_CP), the errors that decide games.

Usage:
  uv run python -m chess_nn.move_quality data/diagnosis_heldout_h1024 --sample 1500 --depth 12 \\
      --runs runs/jj6-h1024-s1000-cp20 runs/rank-a-h512-w0 runs/threat-a-h512-t64-w0
"""

import argparse
import json
import random
import statistics
from multiprocessing import Pool
from pathlib import Path

import chess

from chess_nn.diagnose import BLUNDER_CP, WIN_CP, search
from chess_nn.label import Stockfish
from chess_nn.play import net_move
from chess_nn.rank_pairs import DEFENSIBLE_CP

LOSS_CAP_CP = 1000


def sample_decisions(directory: Path, size: int, seed: int = 0) -> list[dict]:
    decisions = [json.loads(line) for line in (directory / "decisions.jsonl").read_text().splitlines() if line.strip()]
    random.Random(seed).shuffle(decisions)
    return decisions[:size]


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-decision results: each row has best_cp, chosen_cp and agree."""
    n = len(rows)
    losses = [min(max(r["best_cp"] - r["chosen_cp"], 0), LOSS_CAP_CP) for r in rows]
    defensible = [r for r in rows if r["best_cp"] >= DEFENSIBLE_CP]
    turning = sum(1 for r in defensible if r["chosen_cp"] <= -WIN_CP)
    return {
        "decisions": n,
        "agreement": round(sum(r["agree"] for r in rows) / n, 3),
        "mean_loss_cp": round(statistics.mean(losses), 1),
        "large_drop_rate": round(sum(loss >= BLUNDER_CP for loss in losses) / n, 3),
        "defensible_decisions": len(defensible),
        "turning_moves": turning,
        "turning_rate": round(turning / max(len(defensible), 1), 4),
    }


_engine: Stockfish | None = None
_model: Path | None = None
_depth = 12


def _init(model: str, depth: int) -> None:
    global _engine, _model, _depth
    _engine = Stockfish(hash_mb=64)
    _model = Path(model)
    _depth = depth


def _score(decision: dict) -> dict:
    board = chess.Board(decision["fen"])
    chosen = net_move(_model, board).uci()
    if chosen == decision["best"]:
        chosen_cp = decision["best_cp"]
    elif chosen == decision["played"]:
        chosen_cp = decision["played_cp"]
    else:
        history = decision["history"]
        position = f"position fen {decision['start_fen']}" + (f" moves {' '.join(history)}" if history else "")
        chosen_cp = search(_engine, position, _depth, chosen)[0]
    return {"best_cp": decision["best_cp"], "chosen_cp": chosen_cp, "agree": chosen == decision["best"], "chosen": chosen}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("diagnosis", type=Path, help="chess_nn.diagnose output directory of held-out games")
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--sample", type=int, default=1500)
    parser.add_argument("--depth", type=int, default=12, help="match the diagnosis depth")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    decisions = sample_decisions(args.diagnosis, args.sample, args.seed)
    for run in args.runs:
        with Pool(args.workers, initializer=_init, initargs=(str(run / "model.onnx"), args.depth)) as pool:
            rows = pool.map(_score, decisions, chunksize=8)
        result = {"diagnosis": args.diagnosis.name, "depth": args.depth, "seed": args.seed, **summarize(rows)}
        (run / f"move_quality_{args.diagnosis.name}.json").write_text(json.dumps(result, indent=2))
        print(f"{run.name:30s} {json.dumps(result)}")


if __name__ == "__main__":
    main()
