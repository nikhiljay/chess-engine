#!/bin/bash
# Hunt for a model winning every game the grader plays, sampling around a chosen base.
#
#   hunt4.sh <base-run> <seed-start> <seed-end> [parallelism] [sigma-ladder]
#
# A level needs both its games won (ceil(0.7 x 2) = 2 points), so 1.5/2 fails. hunt3 sampled 413
# models around sv-checks and none cleared Level 3: of the 251 that got that far, 133 scored 1.0,
# 4 scored 1.5 and none scored 2.0. sv-checks *loses* its Level 3 White game, and noise at this
# scale does not turn a loss into a win. Bases that only draw the games they miss are a shorter
# hop, so this takes the base as an argument and runs several of them at once.
#
# Torch is pinned to one thread per candidate: unpinned it spawned 60 and the box ran at load 51
# with only 16 candidates in flight, so concurrency was not buying anything.
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1
cd ~/chess-challenge/training
BASE=${1:-runs/sv-both}
START=${2:-1}
END=${3:-4000}
JOBS=${4:-24}
LADDER=${5:-"0.0005 0.001 0.002 0.004 0.008"}
TAG=$(basename $BASE)
COMPETE=~/chess-challenge/target/release/compete
OPENINGS=~/chess-challenge/training/data/startpos_openings.txt
WINNERS=~/winners.txt
read -r -a SIGMAS <<< "$LADDER"

score_of() {
  python3 -c "
import json,sys
print(json.load(open(sys.argv[1]))['levels'][0]['score'])" "$1" 2>/dev/null || echo 0
}

screen_one() {
  local name=$1 sigma=$2 seed=$3
  local dir=/tmp/h4_$name
  mkdir -p $dir
  uv run python -m chess_nn.perturb --from $BASE --name $name --sigma $sigma --seed $seed \
      --runs-dir /tmp/h4_runs > /dev/null 2>&1 || return
  local model=/tmp/h4_runs/$name/model.onnx
  [ -f $model ] || return

  for level in 1 2 3; do
    $COMPETE $model --level $level --openings $OPENINGS --json-output $dir/L$level.json > /dev/null 2>&1
    local s=$(score_of $dir/L$level.json)
    if [ "$s" != "2" ] && [ "$s" != "2.0" ]; then
      [ $level -ge 2 ] && echo "$(date -u +%H:%M) $name reached L$level with $s (sigma $sigma)"
      rm -rf $dir /tmp/h4_runs/$name
      return
    fi
  done
  # Clears L1-L3. Keep it, and record what it does at L4 while we have it.
  $COMPETE $model --level 4 --openings $OPENINGS --json-output $dir/L4.json > /dev/null 2>&1
  local l4=$(score_of $dir/L4.json)
  mkdir -p runs/$name && cp -r /tmp/h4_runs/$name/* runs/$name/
  echo "*** WINNER $name — L1 2 L2 2 L3 2, L4 $l4 (base $TAG, sigma $sigma, seed $seed) ***"
  echo "$name base=$TAG sigma=$sigma seed=$seed L4=$l4" >> $WINNERS
}

mkdir -p /tmp/h4_runs
running=0
for seed in $(seq $START $END); do
  sigma=${SIGMAS[$((seed % ${#SIGMAS[@]}))]}
  screen_one "${TAG}-p${seed}" $sigma $seed &
  running=$((running + 1))
  if [ $running -ge $JOBS ]; then wait -n; running=$((running - 1)); fi
  if [ -s $WINNERS ]; then echo "winner found; stopping"; break; fi
done
wait
echo "hunt4 $TAG $START-$END complete"
