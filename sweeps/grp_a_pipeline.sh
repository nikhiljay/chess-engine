#!/bin/zsh
# grp_a: does penalising error that VARIES BETWEEN SIBLING MOVES make the net play better?
#
# Hypothesis (finding 67/68): the student's harmful error is the part that differs across the
# leaves one decision compares, not its overall bias — Stockfish's static eval carries the same
# won-position bias yet plays 12.6 points better, and a bias shared by every sibling cancels
# when the search picks a move. tact_a already collected exactly those sibling groups; nothing
# in its objective used the grouping, which is why adding the data alone did nothing.
#
# Candidate: the h1024 recipe with tact_a's data plus group_consistency_loss (weights 0.1, 0.5),
# absolute d9 targets unchanged. Controls: the default jj6-h1024-s1000-cp20 (the standing model)
# and tact-a-h1024 (same data, no group term), which isolates the objective from the data.
#
# Steps run one at a time. Exit status lands in runs/grp_a_pipeline.status.
T=/Users/nikhildsouza/workspace/chess-challenge/training
C=$T/runs/jj6-h1024-s1000-cp20
D=$T/runs/tact-a-h1024
cd $T
set -e
trap 'echo $? > $T/runs/grp_a_pipeline.status' EXIT
rm -f $T/runs/grp_a_pipeline.status

# 0. Wait for the group collection to finish, then check it reproduced tact_a's candidate set.
while pgrep -f chess_nn.decision_data > /dev/null; do sleep 30; done
if [[ ! -s $T/data/tactics_groups_a.jsonl ]]; then
  echo "group collection produced no groups file"; exit 1
fi
if cmp -s $T/data/tactics_candidates_a.txt $T/data/tactics_candidates_a_regen.txt; then
  echo "groups: collection reproduced tact_a's candidates exactly"
else
  echo "groups: WARNING regenerated candidates differ from tact_a's; groups still map by FEN"
fi

# 1. Train both weights and play dev L1-L3 (skips finished runs and levels).
uv run --project $T python -m chess_nn.sweep $T/sweeps/grp_a.json > $T/runs/grp_a_sweep.log 2>&1
grep -E 'grp-a|decisions matched' $T/runs/grp_a_sweep.log | tail -5

# 2. Start-position games, reported separately from match strength.
for run in grp-a-h1024-w01 grp-a-h1024-w05; do
  [[ -f $T/runs/$run/startpos_summary.json ]] && continue
  uv run --project $T python -m chess_nn.evaluate $T/runs/$run/model.onnx --levels 1 2 3 \
      --openings $T/data/startpos_openings.txt > $T/runs/$run/startpos_eval.log 2>&1
done

# 3. Paired against the default control, and against same-data-no-group-term.
for run in grp-a-h1024-w01 grp-a-h1024-w05; do
  for control in $C $D; do
    echo "== $run vs ${control:t}"
    uv run --project $T python -m chess_nn.paired games $T/runs/$run=$control --tag dev_100 --levels 1 2 3 | python3 -c "
import json, sys
for level, v in json.load(sys.stdin).items():
    p = v['pairs'][0]; c, k, pr = p['candidate_outcomes'], p['control_outcomes'], p['paired']
    print(' ', level, 'cand', c['score_pct'], c['wdl'], 'rep', c['repetition_draws'], 'rep-win', c['repetition_while_winning'],
          '| ctrl', k['score_pct'], k['wdl'], 'rep', k['repetition_draws'], 'rep-win', k['repetition_while_winning'],
          '| diff', round(pr['mean_points_per_game'], 4), '+-', round(pr['se'], 4), pr['ci95'],
          'better/worse', pr['openings_better'], pr['openings_worse'])
"
  done
done

# 4. Start-position lines, for the record.
uv run --project $T python - <<'EOF'
import json
from pathlib import Path
for run in ['grp-a-h1024-w01', 'grp-a-h1024-w05', 'jj6-h1024-s1000-cp20']:
    for level in (1, 2, 3):
        path = Path(f"/Users/nikhildsouza/workspace/chess-challenge/training/runs/{run}/startpos_L{level}.json")
        if not path.exists():
            continue
        games = json.loads(path.read_text())["levels"][0]["games"]
        print(run, f"startpos L{level}", sum(g["points"] for g in games), [(g["color"], g["points"], g["reason"]) for g in games])
EOF
echo "grp_a pipeline: done"
