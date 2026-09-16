#!/bin/zsh
# Confirmation for comb_a on the fresh openings, which have guided no choice so far.
#
# comb_a (h3072, depth-12 targets, mirror augmentation) is the first candidate to clear the bar:
# pooled over two seeds against seed-matched controls, L3 +0.075 +- 0.025 (CI [+0.026, +0.124]),
# L2 +0.026 +- 0.024, L1 flat. The standing rule is that a credible gain is confirmed on fresh
# games and openings before promotion.
#
# Both candidates and both controls play the same fresh set, then the pooled paired test.
T=/Users/nikhildsouza/workspace/chess-challenge/training
cd $T
set -e
trap 'echo $? > $T/runs/comb_a_fresh.status' EXIT
rm -f $T/runs/comb_a_fresh.status

for run in jj6-h1024-s1000-cp20 rank-b-h1024-w0-seed1 comb-a-h3072-d12-mir comb-a-h3072-d12-mir-seed1; do
  [[ -f $T/runs/$run/fresh_100_summary.json ]] && continue
  uv run --project $T python -m chess_nn.evaluate $T/runs/$run/model.onnx --levels 1 2 3 \
      --openings $T/data/fresh_openings_100.txt > $T/runs/$run/fresh_100_eval.log 2>&1
  echo "$run: fresh openings done"
done

uv run --project $T python -m chess_nn.paired games \
    $T/runs/comb-a-h3072-d12-mir=$T/runs/jj6-h1024-s1000-cp20 \
    $T/runs/comb-a-h3072-d12-mir-seed1=$T/runs/rank-b-h1024-w0-seed1 \
    --tag fresh_100 --levels 1 2 3 | python3 -c "
import json, sys
print('fresh openings, pooled over two seeds:')
for level, v in json.load(sys.stdin).items():
    p = v['pooled']
    print(' ', level, 'pooled', round(p['mean_points_per_game'], 4), '+-', round(p['se'], 4), p['ci95'],
          'better/worse', p['openings_better'], p['openings_worse'])
    for row in v['pairs']:
        print('    ', row['candidate'], row['candidate_outcomes']['score_pct'], row['candidate_outcomes']['wdl'],
              'vs', row['control'], row['control_outcomes']['score_pct'], row['control_outcomes']['wdl'])
"
echo "comb_a fresh: done"
