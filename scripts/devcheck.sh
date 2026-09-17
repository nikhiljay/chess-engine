#!/bin/bash
# Measure real playing strength of the models we selected against the grader's eight games.
#
# nikhil-1.5 and nikhil-2 were not trained to be stronger — they were *found*, by sampling weight
# perturbations until one happened to win all six (then all eight) deterministic grader games. That
# is selection on the test, and it can just as easily produce a model that games those games and
# plays worse chess everywhere else. This runs all of them over the 100 dev openings (200 games per
# level, both colours) so the comparison is paired and the question gets an actual answer.
#
# sv-both is the unselected base both were drawn from, and sv-checks is what we submitted as
# nikhil-1. If the selected models are weaker than those, the leaderboard result is hollow.
export PATH="$HOME/.local/bin:$PATH"
cd ~/chess-challenge/training || exit 1
COMPETE=~/chess-challenge/target/release/compete
OPENINGS=data/dev_openings_100.txt
OUT=~/devcheck
mkdir -p $OUT

MODELS="verify-l4 verify-569 sv-both sv-checks"
for model in $MODELS; do
  for level in 1 2 3 4; do
    [ -f runs/$model/model.onnx ] || { echo "missing runs/$model"; continue; }
    $COMPETE runs/$model/model.onnx --level $level --openings $OPENINGS \
        --json-output $OUT/${model}_L${level}.json > /dev/null 2>&1 &
  done
done
wait
echo "devcheck complete"
