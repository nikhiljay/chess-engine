#!/bin/zsh
# Run one experiment end to end: train the spec's runs, play dev L1-L3, play the
# start-position games separately, pair every run against the standing default, and
# report leaf error where the search looks.
#
#   sweeps/run_experiment.sh <spec.json> <run-name>...
#
# Every step skips finished work, so rerunning continues after an interruption.
# Exit status lands in runs/<spec stem>_pipeline.status.
T=/Users/nikhildsouza/workspace/chess-challenge/training
C=$T/runs/jj6-h1024-s1000-cp20
spec=$1
shift
runs=($@)
name=${${spec:t}:r}
cd $T
set -e
trap 'echo $? > $T/runs/${name}_pipeline.status' EXIT
rm -f $T/runs/${name}_pipeline.status

uv run --project $T python -m chess_nn.sweep $spec > $T/runs/${name}_sweep.log 2>&1
grep -E "^\| ${runs[1]%%-*}-" $T/runs/${name}_sweep.log | tail -${#runs}

for run in $runs; do
  [[ -f $T/runs/$run/startpos_summary.json ]] && continue
  uv run --project $T python -m chess_nn.evaluate $T/runs/$run/model.onnx --levels 1 2 3 \
      --openings $T/data/startpos_openings.txt > $T/runs/$run/startpos_eval.log 2>&1
done

for run in $runs; do
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

uv run --project $T python -m chess_nn.leaf_error $T/runs/teacher/leaves_d9_dev5.jsonl \
    ${runs/#/$T/runs/} $C | grep -E '^[a-z]|level \('

uv run --project $T python - $runs <<'EOF'
import json, sys
from pathlib import Path
root = Path("/Users/nikhildsouza/workspace/chess-challenge/training/runs")
for run in [*sys.argv[1:], "jj6-h1024-s1000-cp20"]:
    for level in (1, 2, 3):
        path = root / run / f"startpos_L{level}.json"
        if path.exists():
            games = json.loads(path.read_text())["levels"][0]["games"]
            print(run, f"startpos L{level}", sum(g["points"] for g in games), [(g["color"], g["points"], g["reason"]) for g in games])
EOF
echo "$name pipeline: done"
