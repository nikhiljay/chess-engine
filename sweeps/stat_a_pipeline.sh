#!/bin/zsh
# stat_a: distil Stockfish's STATIC evaluation instead of its depth-9 search score.
#
# Why the target changes (findings 67-73):
#  - A leaf value in this harness must say what the position is worth as it stands. The search
#    resolves captures itself, so a searched target is counted twice; leaf_a showed that costing
#    21 points at L2, and 56% of the search's leaves fail the quiet-best-move filter.
#  - Stockfish's static eval is the function that plays this exact role inside Stockfish's own
#    search, and as a leaf function in the teacher test it scored 68.8% against our net's 56.2%.
#  - It is deterministic, so the labels carry no search noise, and it needs no filtering.
#
# Data: the 1.5M leaves the search actually evaluated, plus 2M human positions for coverage,
# all labeled with the static eval. Waits for any running labeler so the cores are not split.
T=/Users/nikhildsouza/workspace/chess-challenge/training
cd $T
set -e
trap 'echo $? > $T/runs/stat_a_pipeline.status' EXIT
rm -f $T/runs/stat_a_pipeline.status

while pgrep -f 'chess_nn.label --candidates' > /dev/null; do sleep 30; done

if [[ ! -f $T/data/shards_static/shard_static_leaf_0000.npz ]]; then
  uv run --project $T python -m chess_nn.static_label --candidates $T/data/leaf_candidates_a.txt \
      --out-dir $T/data/shards_static --prefix shard_static_leaf --workers 10 \
      >> $T/data/label_static.log 2>&1
fi
if [[ ! -f $T/data/shards_static/shard_static_human_0000.npz ]]; then
  uv run --project $T python -m chess_nn.static_label --candidates $T/data/candidates.txt \
      --out-dir $T/data/shards_static --prefix shard_static_human --limit 2000000 --workers 10 \
      >> $T/data/label_static.log 2>&1
fi
tail -3 $T/data/label_static.log

exec /bin/zsh $T/sweeps/run_experiment.sh $T/sweeps/stat_a.json stat-a-h1024
