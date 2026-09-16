#!/bin/zsh
# Second overnight queue, after mir_a and rep_a.
#
# deep_a found depth-12 targets ahead of depth-9 targets on the same 2M positions by
# +0.055 +- 0.042 at L2 (46 openings better, 32 worse) — positive but unresolved, which by the
# standing rule earns replication rather than promotion. Two steps:
#
#  1. deep_a_rep — the same comparison at seed 1, to pool and halve the interval.
#  2. deep_b — depth 12 at full scale (both label batches, ~4.3M positions) at two seeds,
#     against the default. This is the test that matters: the default's advantage in deep_a
#     was partly its 4.36M positions against the arms' 2M.
#
# Waits for the depth-12 labeling of the second batch, and for the machine to be quiet.
T=/Users/nikhildsouza/workspace/chess-challenge/training
cd $T
trap 'echo $? > $T/runs/overnight_queue2.status' EXIT
rm -f $T/runs/overnight_queue2.status

wait_for_quiet() {
  while pgrep -f 'run_experiment.sh|chess_nn.sweep' > /dev/null; do sleep 60; done
}

# The first queue still has mir_a and rep_a to run; do not interleave with it.
while pgrep -f 'overnight_queue.sh' > /dev/null; do sleep 120; done

wait_for_quiet
/bin/zsh $T/sweeps/run_experiment.sh $T/sweeps/deep_a_rep.json deep-a-d12-seed1 deep-a-d9-seed1 \
    > $T/runs/deep_a_rep_pipeline.log 2>&1 || echo "deep_a_rep failed with $?"

# Full-scale depth 12 needs the second label batch finished.
while pgrep -f 'chess_nn.label --candidates' > /dev/null; do sleep 120; done
wait_for_quiet
/bin/zsh $T/sweeps/run_experiment.sh $T/sweeps/deep_b.json deep-b-d12-full deep-b-d12-full-seed1 \
    > $T/runs/deep_b_pipeline.log 2>&1 || echo "deep_b failed with $?"

echo "overnight queue 2: done"
