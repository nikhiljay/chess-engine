#!/bin/zsh
# out_a: label positions with the GAME RESULT against these baseline bots.
#
# Everything so far distils a Stockfish score. Findings 73-74 showed that target can mislead:
# the search resolves captures itself, and the most faithful distillation of a static evaluation
# played 12 points worse than the default. What we actually measure is games won against these
# bots, and an outcome label optimises that directly — including the shallow search's quirks,
# the opponent's habits, and the fact that a repetition draw is worth half a point.
#
# 15,000 games from the default model at Levels 1-3 (datagen's random openings, no random moves),
# blended with the depth-12 mix so general evaluation knowledge is not lost. One bit per game is
# noisy, hence the volume. Two seeds, since single arms in this project are noise.
T=/Users/nikhildsouza/workspace/chess-challenge/training
M=$T/runs/jj6-h1024-s1000-cp20
cd $T
set -e
trap 'echo $? > $T/runs/out_a_pipeline.status' EXIT
rm -f $T/runs/out_a_pipeline.status

while pgrep -f 'run_experiment.sh|chess_nn.sweep' > /dev/null; do sleep 60; done

for level in 1 2 3; do
  [[ -s $T/data/outcome_games_L${level}.jsonl ]] && continue
  $T/datagen/target/release/datagen $M/model.onnx --out $T/data/outcome_positions_L${level}.txt \
      --games 5000 --threads 10 --level $level --random-move-prob 0 --sample-prob 0 \
      --seed 90$level --record $T/data/outcome_games_L${level}.jsonl >> $T/data/outcome_games.log 2>&1
  echo "level $level games: $(wc -l < $T/data/outcome_games_L${level}.jsonl)"
done

if [[ ! -f $T/data/shards_outcome/shard_outcome_a_0000.npz ]]; then
  uv run --project $T python -m chess_nn.outcome_shard $T/data/outcome_games_L1.jsonl \
      $T/data/outcome_games_L2.jsonl $T/data/outcome_games_L3.jsonl \
      --out-dir $T/data/shards_outcome --prefix shard_outcome_a
fi

exec /bin/zsh $T/sweeps/run_experiment.sh $T/sweeps/out_a.json out-a-h3072-mix out-a-h3072-mix-seed1
