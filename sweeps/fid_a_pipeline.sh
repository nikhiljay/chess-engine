#!/bin/zsh
# fid_a: does weighting ABSOLUTE accuracy harder improve play?
#
# Findings 67-69: the search wins 94% of these games when leaf values are accurate, the net's
# error is 4-6 pawns exactly where the search spends over half its leaves (won positions), and
# both objectives that traded absolute accuracy for relative structure lost badly. The centipawn
# Huber term is the one that cares about absolute pawns; nothing above weight 0.2 has been tried
# at this recipe. Scale 1500 is included because a flatter sigmoid keeps resolution between large
# advantages, which is the regime at issue.
#
# Data is the broad mix only, so the objective is the single change from the default control.
# Exit status lands in runs/fid_a_pipeline.status.
T=/Users/nikhildsouza/workspace/chess-challenge/training
C=$T/runs/jj6-h1024-s1000-cp20
RUNS=(fid-a-h1024-cp05 fid-a-h1024-cp10 fid-a-h1024-cp10-s1500)
cd $T
set -e
trap 'echo $? > $T/runs/fid_a_pipeline.status' EXIT
rm -f $T/runs/fid_a_pipeline.status

# 1. Train and play dev L1-L3 (skips finished runs and levels).
uv run --project $T python -m chess_nn.sweep $T/sweeps/fid_a.json > $T/runs/fid_a_sweep.log 2>&1
grep -E 'fid-a' $T/runs/fid_a_sweep.log | tail -4

# 2. Start-position games, kept separate from match strength.
for run in $RUNS; do
  [[ -f $T/runs/$run/startpos_summary.json ]] && continue
  uv run --project $T python -m chess_nn.evaluate $T/runs/$run/model.onnx --levels 1 2 3 \
      --openings $T/data/startpos_openings.txt > $T/runs/$run/startpos_eval.log 2>&1
done

# 3. Paired against the standing default.
for run in $RUNS; do
  echo "== $run vs jj6-h1024-s1000-cp20"
  uv run --project $T python -m chess_nn.paired games $T/runs/$run=$C --tag dev_100 --levels 1 2 3 | python3 -c "
import json, sys
for level, v in json.load(sys.stdin).items():
    p = v['pairs'][0]; c, k, pr = p['candidate_outcomes'], p['control_outcomes'], p['paired']
    print(' ', level, 'cand', c['score_pct'], c['wdl'], 'rep', c['repetition_draws'], 'rep-win', c['repetition_while_winning'],
          '| ctrl', k['score_pct'], k['wdl'], 'rep', k['repetition_draws'], 'rep-win', k['repetition_while_winning'],
          '| diff', round(pr['mean_points_per_game'], 4), '+-', round(pr['se'], 4), pr['ci95'],
          'better/worse', pr['openings_better'], pr['openings_worse'])
"
done

# 4. Mechanism check: leaf error where the search looks (must fall in the winning buckets).
uv run --project $T python -m chess_nn.leaf_error $T/runs/teacher/leaves_d9_dev5.jsonl \
    $C $T/runs/fid-a-h1024-cp05 $T/runs/fid-a-h1024-cp10 $T/runs/fid-a-h1024-cp10-s1500

# 5. Start-position lines, for the record.
uv run --project $T python - <<'EOF'
import json
from pathlib import Path
root = Path("/Users/nikhildsouza/workspace/chess-challenge/training/runs")
for run in ['fid-a-h1024-cp05', 'fid-a-h1024-cp10', 'fid-a-h1024-cp10-s1500', 'jj6-h1024-s1000-cp20']:
    for level in (1, 2, 3):
        path = root / run / f"startpos_L{level}.json"
        if path.exists():
            games = json.loads(path.read_text())["levels"][0]["games"]
            print(run, f"startpos L{level}", sum(g["points"] for g in games), [(g["color"], g["points"], g["reason"]) for g in games])
EOF
echo "fid_a pipeline: done"
