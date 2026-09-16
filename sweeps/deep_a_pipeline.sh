#!/bin/zsh
# deep_a: does a DEEPER target play better?
#
# Finding 74's explanation for tonight's inversions: a searched target carries lookahead this
# search cannot perform (depth 1 plus captures, no check extensions), so the net substitutes for
# search depth. Distilling Stockfish's static eval produced the most accurate evaluator of the
# project and played 12 points worse at L2, while the d9-trained default plays best. If that is
# right, depth 12 targets should beat depth 9 targets.
#
# Controlled: the same 2M positions, relabeled at depth 12; the d9 arm is restricted to exactly
# the subset depth 12 kept, so label depth is the only difference between the arms.
T=/Users/nikhildsouza/workspace/chess-challenge/training
cd $T
set -e
trap 'echo $? > $T/runs/deep_a_pipeline.status' EXIT
rm -f $T/runs/deep_a_pipeline.status

while pgrep -f 'chess_nn.label --candidates' > /dev/null; do sleep 60; done
tail -1 $T/data/label_d12.log

if [[ ! -f $T/data/shards_d9_matched/shard_d9m_0000.npz ]]; then
  uv run --project $T python -m chess_nn.match_shard --like $T/data/shards_d12 \
      --from $T/data/shards/shard_0000.npz $T/data/shards/shard_0001.npz $T/data/shards/shard_0002.npz \
             $T/data/shards/shard_0003.npz $T/data/shards/shard_2017-07_0000.npz $T/data/shards/shard_2017-07_0001.npz \
      --out-dir $T/data/shards_d9_matched --prefix shard_d9m
fi

exec /bin/zsh $T/sweeps/run_experiment.sh $T/sweeps/deep_a.json deep-a-d12 deep-a-d9
