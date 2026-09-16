#!/bin/zsh
# One bounded cycle on the only lever with a large measured effect on play: data volume.
#
# Evidence: the same h1024 recipe scores L2 39.2% on 1.5M positions and 54.2% on 4.36M. Every
# other lever tried moved L2 by at most a few points, and most hurt. 9M candidates were already
# sampled, so this is labeling plus one training run, not a new idea.
#
# The default (seed 0) and its twin rank-b-h1024-w0-seed1 are the same recipe at 4.36M and the
# same epochs, so they are exact controls: data volume is the only change.
T=/Users/nikhildsouza/workspace/chess-challenge/training
cd $T
trap 'echo $? > $T/runs/scale_cycle.status' EXIT
rm -f $T/runs/scale_cycle.status

# 1. Let the architecture screen finish before taking the machine.
while pgrep -f 'run_experiment.sh.*arch_b' > /dev/null; do sleep 120; done

# 2. A confirmation opening set that has guided nothing (the previous fresh set is spent).
if [[ ! -s $T/data/fresh_openings_d_100.txt ]]; then
  nice -n 10 uv run --project $T python -m chess_nn.dev_openings --pgn $T/data/lichess_2017-07.pgn.zst \
      --out $T/data/fresh_openings_d_100.txt --count 100 --seed 11 \
      --exclude $T/data/dev_openings_100.txt $T/data/fresh_openings_100.txt \
      > $T/data/fresh_openings_d.log 2>&1 || echo "opening generation failed"
fi

# 3. Wait for labeling to reach the target volume (11 shards ~ 8.5M positions), then stop it so
#    the evaluation games get the cores back.
while [[ $(ls $T/data/shards/*.npz 2>/dev/null | wc -l) -lt 11 ]] && pgrep -f 'chess_nn.label' > /dev/null; do sleep 180; done
pkill -f label_more.sh; pkill -f 'chess_nn.label'
sleep 5
uv run --project $T python -c "
from pathlib import Path
from chess_nn.shards import load_shards
from chess_nn.sweep import expand_shards
_, scores, _ = load_shards([Path(p) for p in expand_shards(['data/shards/shard_*.npz'])])
print(f'training mix: {len(scores):,} positions')
"

exec /bin/zsh $T/sweeps/run_experiment.sh $T/sweeps/scale_a.json scale-a-h1024-full scale-a-h1024-full-seed1
