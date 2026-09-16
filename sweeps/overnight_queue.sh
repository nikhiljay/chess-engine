#!/bin/zsh
# Overnight queue: one experiment at a time, so nothing contends for cores or memory.
#
#  1. mir_a — mirror augmentation, against the overfitting measured in finding 72
#     (h1024 train/val 0.00239/0.00287, h3072 worse), at two widths since the bigger net
#     overfits harder and may only pay off once the data is effectively doubled.
#  2. rep_a — two more seeds of the best non-default arm so far (scale 1500, cp weight 1.0,
#     which scored L2 56.0 at seed 0). Seed spread is about +-2 points, so one arm cannot be
#     told from the default without replication.
#
# Labeling is deliberately NOT waited on: it is nice'd and CPU-bound while training uses the
# GPU, so the depth-12 batch runs alongside. Only trainings and evaluations are serialised.
# Each step skips finished runs, so rerunning resumes. Status in runs/overnight_queue.status.
T=/Users/nikhildsouza/workspace/chess-challenge/training
cd $T
trap 'echo $? > $T/runs/overnight_queue.status' EXIT
rm -f $T/runs/overnight_queue.status

wait_for_quiet() {
  while pgrep -f 'run_experiment.sh|chess_nn.sweep' > /dev/null; do sleep 60; done
}

wait_for_quiet
/bin/zsh $T/sweeps/run_experiment.sh $T/sweeps/mir_a.json mir-a-h1024 mir-a-h3072 \
    > $T/runs/mir_a_pipeline.log 2>&1 || echo "mir_a failed with $?"

wait_for_quiet
/bin/zsh $T/sweeps/run_experiment.sh $T/sweeps/rep_a.json rep-a-s1500-seed1 rep-a-s1500-seed2 \
    > $T/runs/rep_a_pipeline.log 2>&1 || echo "rep_a failed with $?"

echo "overnight queue: done"
