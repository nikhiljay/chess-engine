#!/bin/zsh
# dis_a: stop the net being over-confident in level positions.
#
# Finding 70: on the leaves the search evaluates, in level positions (|d9| <= 3, where games are
# decided) the truth has sd 1.42 and Stockfish's static eval predicts with sd 1.73, while every
# net of ours predicts with sd ~3.26 and correlates far worse (r 0.35 vs 0.56). They are badly
# wrong (>3 pawns) on 29.6% of those leaves against static's 9.9% — and a constant 0 beats them
# all on MAE there. An over-confident leaf value is exactly what makes the search choose it.
#
# The Huber term is linear past one pawn, so those big misses are cheap. These arms price them
# quadratically (mse), and/or narrow the target range so the net spans less.
# Exit status lands in runs/dis_a_pipeline.status.
T=/Users/nikhildsouza/workspace/chess-challenge/training
C=$T/runs/jj6-h1024-s1000-cp20
RUNS=(dis-a-mse dis-a-mse-w10 dis-a-mse-clamp1000 dis-a-huber-clamp1000)
cd $T
set -e
trap 'echo $? > $T/runs/dis_a_pipeline.status' EXIT
rm -f $T/runs/dis_a_pipeline.status

uv run --project $T python -m chess_nn.sweep $T/sweeps/dis_a.json > $T/runs/dis_a_sweep.log 2>&1
grep -E '^\| dis-a' $T/runs/dis_a_sweep.log | tail -4

for run in $RUNS; do
  [[ -f $T/runs/$run/startpos_summary.json ]] && continue
  uv run --project $T python -m chess_nn.evaluate $T/runs/$run/model.onnx --levels 1 2 3 \
      --openings $T/data/startpos_openings.txt > $T/runs/$run/startpos_eval.log 2>&1
done

for run in $RUNS; do
  echo "== $run vs jj6-h1024-s1000-cp20"
  uv run --project $T python -m chess_nn.paired games $T/runs/$run=$C --tag dev_100 --levels 1 2 3 | python3 -c "
import json, sys
for level, v in json.load(sys.stdin).items():
    p = v['pairs'][0]; c, k, pr = p['candidate_outcomes'], p['control_outcomes'], p['paired']
    print(' ', level, 'cand', c['score_pct'], c['wdl'], 'rep', c['repetition_draws'],
          '| ctrl', k['score_pct'], k['wdl'], 'rep', k['repetition_draws'],
          '| diff', round(pr['mean_points_per_game'], 4), '+-', round(pr['se'], 4), pr['ci95'],
          'better/worse', pr['openings_better'], pr['openings_worse'])
"
done

# Mechanism check: level-position spread and big-error rate must fall.
uv run --project $T python -m chess_nn.leaf_error $T/runs/teacher/leaves_d9_dev5.jsonl $C \
    $T/runs/dis-a-mse $T/runs/dis-a-mse-w10 $T/runs/dis-a-mse-clamp1000 $T/runs/dis-a-huber-clamp1000 \
    2>&1 | grep -E '^[a-z]|level \('

uv run --project $T python - <<'EOF'
import json
from pathlib import Path
root = Path("/Users/nikhildsouza/workspace/chess-challenge/training/runs")
for run in ['dis-a-mse', 'dis-a-mse-w10', 'dis-a-mse-clamp1000', 'dis-a-huber-clamp1000', 'jj6-h1024-s1000-cp20']:
    for level in (1, 2, 3):
        path = root / run / f"startpos_L{level}.json"
        if path.exists():
            games = json.loads(path.read_text())["levels"][0]["games"]
            print(run, f"startpos L{level}", sum(g["points"] for g in games), [(g["color"], g["points"], g["reason"]) for g in games])
EOF
echo "dis_a pipeline: done"
