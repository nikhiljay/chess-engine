#!/bin/zsh
# ep_a: does LESS training play better?
#
# comb_b gave a clean controlled comparison — same width, same data, same seed, 20 epochs
# against 30 — and 20 epochs won L2 by 4.3 points (57.5 against 53.2). That is finding 74 once
# more: fitting the Stockfish target harder does not play better, because the target's value is
# the lookahead it carries, not its accuracy. If so, stopping earlier should help again.
#
# 10 epochs at two seeds, against comb_a's 20-epoch runs at the same seeds and the default.
T=/Users/nikhildsouza/workspace/chess-challenge/training
cd $T
while pgrep -f 'run_experiment.sh|chess_nn.sweep|datagen ' > /dev/null; do sleep 60; done
exec /bin/zsh $T/sweeps/run_experiment.sh $T/sweeps/ep_a.json ep-a-h3072-e10 ep-a-h3072-e10-seed1
