"""Paired comparisons of candidate runs against matched controls.

games: compete plays every opening with both colours in a fixed order, so two
runs' games on the same openings pair up by index (checked by starting position
and colour). Per level, and pooled over all given candidate=control pairs
(e.g. one per seed), it reports the mean per-game point difference with a
standard error that treats each opening's two games as one unit, a 95%
interval, how many openings got better or worse, W/D/L, repetition draws, and
repetition draws the net was winning.

decisions: chess_nn.move_quality row files for the same decision sample pair up
by (game, ply). It counts turning moves — from a defensible position to a
losing one — that the candidate prevented or introduced relative to the
control, with an exact two-sided sign test, and the paired change in mean loss.

Usage:
  uv run python -m chess_nn.paired games runs/rank-b-h1024-w005=runs/jj6-h1024-s1000-cp20 \\
      runs/rank-b-h1024-w005-seed1=runs/rank-b-h1024-w0-seed1 --tag dev_100
  uv run python -m chess_nn.paired decisions runs/rank-b-h1024-w005=runs/jj6-h1024-s1000-cp20 --quality diagnosis_fresh_h1024
"""

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from chess_nn.diagnose import WIN_CP
from chess_nn.move_quality import LOSS_CAP_CP
from chess_nn.rank_pairs import DEFENSIBLE_CP


def load_games(run: Path, tag: str, level: int) -> list[dict]:
    return json.loads((run / f"{tag}_L{level}.json").read_text())["levels"][0]["games"]


def opening_differences(candidate: list[dict], control: list[dict]) -> list[float]:
    """Point difference (candidate − control) summed over each opening's games."""
    if len(candidate) != len(control):
        raise ValueError(f"game counts differ: {len(candidate)} vs {len(control)}")
    by_opening: dict[str, float] = defaultdict(float)
    for a, b in zip(candidate, control):
        if a["starting_fen"] != b["starting_fen"] or a["color"] != b["color"]:
            raise ValueError("games are not on the same openings in the same order")
        by_opening[a["starting_fen"]] += a["points"] - b["points"]
    return list(by_opening.values())


def interval(opening_diffs: list[float], games_per_opening: float) -> dict:
    n = len(opening_diffs)
    mean = statistics.mean(opening_diffs) / games_per_opening
    se = statistics.stdev(opening_diffs) / math.sqrt(n) / games_per_opening if n > 1 else float("nan")
    return {
        "openings": n,
        "mean_points_per_game": round(mean, 4),
        "se": round(se, 4),
        "ci95": [round(mean - 1.96 * se, 4), round(mean + 1.96 * se, 4)],
        "openings_better": sum(d > 0 for d in opening_diffs),
        "openings_worse": sum(d < 0 for d in opening_diffs),
    }


def outcomes(games: list[dict]) -> dict:
    points = [g["points"] for g in games]
    return {
        "score_pct": round(100 * sum(points) / len(points), 1),
        "wdl": f"{sum(p == 1.0 for p in points)}/{sum(p == 0.5 for p in points)}/{sum(p == 0.0 for p in points)}",
        "repetition_draws": sum(g["reason"] == "repetition" for g in games),
    }


def compare_games(pairs: list[tuple[Path, Path]], tag: str, levels: list[int]) -> dict:
    report = {}
    for level in levels:
        pooled: list[float] = []
        rows = []
        for candidate, control in pairs:
            a, b = load_games(candidate, tag, level), load_games(control, tag, level)
            diffs = opening_differences(a, b)
            pooled += diffs
            summaries = {run.name: {s["level"]: s for s in json.loads((run / f"{tag}_summary.json").read_text())} for run in (candidate, control)}
            rows.append({
                "candidate": candidate.name,
                "control": control.name,
                "candidate_outcomes": {**outcomes(a), "repetition_while_winning": summaries[candidate.name][level].get("repetition_while_winning")},
                "control_outcomes": {**outcomes(b), "repetition_while_winning": summaries[control.name][level].get("repetition_while_winning")},
                "paired": interval(diffs, games_per_opening=len(a) / len(diffs)),
            })
        games_per_opening = 2.0
        report[f"L{level}"] = {"pairs": rows, "pooled": interval(pooled, games_per_opening)}
    return report


def sign_test_p(successes: int, failures: int) -> float:
    """Exact two-sided binomial test of successes vs failures at p = 0.5."""
    n = successes + failures
    if n == 0:
        return 1.0
    k = min(successes, failures)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def turning(row: dict) -> bool:
    return row["best_cp"] >= DEFENSIBLE_CP and row["chosen_cp"] <= -WIN_CP


def compare_decisions(candidate_rows: list[dict], control_rows: list[dict]) -> dict:
    control = {(r["game"], r["ply"]): r for r in control_rows}
    matched = [(r, control[(r["game"], r["ply"])]) for r in candidate_rows if (r["game"], r["ply"]) in control]
    prevented = sum(turning(b) and not turning(a) for a, b in matched)
    introduced = sum(turning(a) and not turning(b) for a, b in matched)
    loss = lambda r: min(max(r["best_cp"] - r["chosen_cp"], 0), LOSS_CAP_CP)
    loss_diffs = [loss(a) - loss(b) for a, b in matched]
    se = statistics.stdev(loss_diffs) / math.sqrt(len(loss_diffs)) if len(loss_diffs) > 1 else float("nan")
    changed = Counter("same move" if a["chosen"] == b["chosen"] else "different move" for a, b in matched)
    return {
        "decisions": len(matched),
        "choices": dict(changed),
        "turning_candidate": sum(turning(a) for a, _ in matched),
        "turning_control": sum(turning(b) for _, b in matched),
        "turning_prevented": prevented,
        "turning_introduced": introduced,
        "sign_test_p": round(sign_test_p(prevented, introduced), 4),
        "mean_loss_diff_cp": round(statistics.mean(loss_diffs), 2) if loss_diffs else None,
        "mean_loss_diff_se": round(se, 2),
    }


def parse_pairs(specs: list[str]) -> list[tuple[Path, Path]]:
    return [tuple(Path(side) for side in spec.split("=", 1)) for spec in specs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    g = commands.add_parser("games")
    g.add_argument("pairs", nargs="+", help="CANDIDATE_RUN=CONTROL_RUN")
    g.add_argument("--tag", default="dev_100")
    g.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    d = commands.add_parser("decisions")
    d.add_argument("pairs", nargs="+", help="CANDIDATE_RUN=CONTROL_RUN")
    d.add_argument("--quality", required=True, help="move_quality diagnosis name")
    args = parser.parse_args()

    if args.command == "games":
        print(json.dumps(compare_games(parse_pairs(args.pairs), args.tag, args.levels), indent=2))
        return
    for candidate, control in parse_pairs(args.pairs):
        rows = lambda run: [json.loads(line) for line in (run / f"move_quality_{args.quality}_rows.jsonl").read_text().splitlines() if line.strip()]
        print(candidate.name, "vs", control.name, json.dumps(compare_decisions(rows(candidate), rows(control))))


if __name__ == "__main__":
    main()
