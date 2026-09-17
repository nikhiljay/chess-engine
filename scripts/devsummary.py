"""Compare dev strength of the grader-selected models against their unselected base.

nikhil-1.5 (verify-569) and nikhil-2 (verify-l4) were found by sampling weight perturbations of
sv-both until one won all the grader's deterministic games. That is selection on the test. The
question this answers is whether the selection cost real playing strength.

Reports each model's score over the 200 dev games per level, and the paired per-opening difference
against the base, which is the comparison that actually controls for opening difficulty.
"""

import json
import pathlib
import sys
from collections import defaultdict

OUT = pathlib.Path.home() / "devcheck"
BASE = "sv-both"
LEVELS = (1, 2, 3, 4)


def load(model: str, level: int):
    path = OUT / f"{model}_L{level}.json"
    if not path.exists():
        return None
    level_result = json.loads(path.read_text())["levels"][0]
    # Key each game by (opening, colour) so two models can be compared game for game.
    return {(g["opening_index"], g["color"]): g["points"] for g in level_result["games"]}


def main() -> None:
    models = sys.argv[1:] or ["verify-l4", "verify-569", "sv-both", "sv-checks"]
    per_level = defaultdict(dict)
    for model in models:
        for level in LEVELS:
            games = load(model, level)
            if games:
                per_level[level][model] = games

    print(f"{'level':>6}  {'model':<12} {'games':>6} {'score':>8} {'pct':>7}   vs base (paired)")
    for level in LEVELS:
        entries = per_level.get(level)
        if not entries:
            continue
        base = entries.get(BASE)
        for model in models:
            games = entries.get(model)
            if not games:
                continue
            total = sum(games.values())
            pct = 100.0 * total / len(games)
            delta = ""
            if base and model != BASE:
                shared = set(games) & set(base)
                diff = sum(games[k] - base[k] for k in shared)
                better = sum(1 for k in shared if games[k] > base[k])
                worse = sum(1 for k in shared if games[k] < base[k])
                delta = f"{diff:+6.1f} pts over {len(shared)} games  (+{better}/-{worse})"
            print(f"{level:>6}  {model:<12} {len(games):>6} {total:>8.1f} {pct:>6.1f}%   {delta}")
        print()


if __name__ == "__main__":
    main()
