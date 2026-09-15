"""Summarize every run: model shape, data size, losses and match results.

Validation losses are only comparable between runs with the same WDL scale.

Usage:
  uv run python -m chess_nn.report
"""

import json
from pathlib import Path

from chess_nn.harness import RUNS_DIR


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def collect_runs(runs_dir: Path = RUNS_DIR) -> list[dict]:
    rows = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        config = _load(run_dir / "config.json") or {}
        summary = _load(run_dir / "summary.json") or {}
        results = {}
        for path in sorted(run_dir.glob("*_summary.json")):
            tag = path.name.removesuffix("_summary.json")
            results[tag] = {entry["level"]: entry for entry in json.loads(path.read_text())}
        rows.append({
            "run": run_dir.name,
            "params": summary.get("params"),
            "positions": summary.get("positions"),
            "hidden": config.get("hidden"),
            "head": config.get("head", [32, 32]),
            "wdl_scale_cp": config.get("wdl_scale_cp", 400.0),
            "val_loss": summary.get("final_val_loss"),
            "results": results,
        })
    return rows


def _fmt(value, spec: str) -> str:
    return "?" if value is None else format(value, spec)


def format_table(rows: list[dict]) -> str:
    lines = [
        "| run | params | positions | H | head | scale | val loss | results |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        results = "; ".join(
            f"{tag}: " + " ".join(
                f"L{level} {entry['score_pct']:.0f}%{' PASS' if entry['passed'] else ''}"
                for level, entry in sorted(levels.items())
            )
            for tag, levels in row["results"].items()
        )
        head = "x".join(map(str, row["head"])) or "none"
        lines.append(
            f"| {row['run']} | {_fmt(row['params'], ',')} | {_fmt(row['positions'], ',')} | "
            f"{row['hidden']} | {head} | {row['wdl_scale_cp']:g} | {_fmt(row['val_loss'], '.5f')} | {results} |"
        )
    return "\n".join(lines)


def main() -> None:
    print(format_table(collect_runs()))


if __name__ == "__main__":
    main()
