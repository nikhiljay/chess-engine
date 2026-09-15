# Overnight log — 2026-09-14

Goal: beat leaderboard #1 `st-h128-s800-9M` (99,358 params, passes Level 2) —
pass Level 3, or pass Level 2 with fewer params. Iteration uses dev openings
only; official openings only to confirm a milestone.

## Findings so far

All dev numbers are `data/dev_openings_100.txt` (200 games per level) unless
noted. Full table: `uv run python -m chess_nn.report`.

1. **More data helps.** M1 (109k positions) 41% L1 on the 50-opening set;
   same net on June shard 0 (726k) 56% on that set.
2. **Shard-0 sweep (726k positions):** h256-s400 L1 61% / L2 38%;
   h256-s800 57/41; h128-s400 56/41; h128-s800 50/31; h128-none-s800 54/32.
   WDL scale 800 gave no clear gain at this data size.
3. **Repetition is the dominant leak.** ~45% of games end in threefold
   repetition. For h256-s400, 108 of 400 games were repetition draws with the
   net ahead by its own eval; Stockfish depth 14 confirms 94 above +2 and 81
   above +5 (8 forced mates). Nearly all are two-move shuffles (a piece moves
   back and forth while the opponent shuffles), across all game phases.
   Depth-1 search cannot see mate-in-2, so the eval itself must rank
   "closer to mate" higher.
4. **Mate ladder on human data: no gain.** Only 2.0% of human positions are
   mates, so targets barely change; L1 still moved 9 points between
   near-identical runs, so run-to-run noise is large.
5. **Seed noise is ~±3 points.** h256-s400 on shard 0 with seeds 0/1/2:
   L1 60.8 / 54.8 / 55.5%, L2 37.8 / 41.0 / 42.5%. Most differences in
   findings 2 and 4 are within this noise; only h128-s800 looks clearly worse.
   Conclusion: at 726k human positions, architecture/scale/ladder choices
   barely matter; the data and the repetition leak do.
6. **June shards 0-3 (2.9M positions):** h256-s400 L1 64.0% / L2 45.0%,
   +7 / +5 over the shard-0 seed mean, so more data does help the 215k net;
   h128-s400 56.5% / 41.8%, no gain, so width matters once data grows.
   Train/val gap nearly closed (0.0087 / 0.0092). Repetition still ~38% of
   games, with 56 (L1) and 37 (L2) of them the net ahead by its own eval.
7. **Capacity gains get absorbed by repetition.** On June shards 0-3:
   h256 seed1 L1 63.7% / L2 46.0%; h512 (429k params) 63.7 / 49.8;
   h256 40 epochs 63.2 / 47.2. Bigger or longer-trained nets cut losses
   (L1 34 → 21-25, L2 71 → 49-55) but those games become repetition draws
   (L1 76 → 95-103) with the net ahead in 64-71 of them. L1 is capped near
   64% until conversion is fixed; h512 converting its "winning" repetitions
   would score ~81%.
8. **Progress shaping: no improvement.** `chess_nn/shaping.py` adds a
   winning-side bonus to non-mate targets with |score| ≥ 300 cp: 10 cp per
   pawn rank advanced, 4 cp per point of material traded off, and 15 cp per
   step of mop-up (losing king to the edge, kings close) once the loser has
   no pawns and ≤ a rook. h256 on shards 0-3: s400 L1 62.7% / L2 43.0% with
   repetition-while-winning rising to 75 / 68; s1000 61.0 / 45.5. The two
   positions in a shuffle loop have identical progress features, so the
   bonus cannot break the tie, and 10-15 cp sits inside the net's eval error.
9. **Loop diagnosis (h512, shards 0-3).** At loop positions the net's scores
   for its top moves are nearly flat in won endgames: KR+P vs K with mate in 5,
   Rg7 +8.61 / Rf7 +8.60 / h7 +8.59 / Rb7 +8.50, while Stockfish sees mate in
   4-7 for three of them and +14 for the fourth. It also sometimes prefers moves Stockfish rates as
   draws (Re5: net +7.31, Stockfish 0). The WDL loss at scale 400 maps +8 and
   mate to nearly the same win probability, so the net gets almost no gradient
   to rank them. (Caveat: children were ranked by static eval; the harness
   uses quiescence, so capture moves are overrated in that listing.)
   Next test: `sweeps/june4_selfplay.json` — shards 0-3 plus self-play
   positions, with the mate ladder and an auxiliary Huber loss in pawns
   (`cp_loss_weight`) that keeps gradient between large advantages.
10. **Self-play data (`data/shards_selfplay/shard_selfplay_a_0000.npz`):**
    154,944 positions kept of 212,568 (depth 9, depth-14 re-search at
    |score| ≥ 600, ~50 positions/s with 6 workers). 14.3% are mate scores
    (human data: 2.0%), 44% are non-mate |score| ≥ 600, 77% are |score| ≥ 300,
    i.e. heavily weighted toward won positions, as intended.
11. **Self-play positions cut losses, not repetition.** h256-s400 on shard 0 +
    self-play (881k, 18% self-play): L1 64.0% / L2 38.2% with 22 L1 losses
    (shard-0 seeds: 35-43 losses, L1 mean 57.0, L2 mean 40.4); with the mate
    ladder 62.0 / 39.2. L1 gains +5-7 over the seed mean, beyond noise, but
    repetition-while-winning stays at 57-62 and L2 does not move. The mate
    ladder shows no effect even with 14% mates in the self-play data.
12. **Shards 0-3 + self-play (3.06M, 5% self-play), h256-s400:** plain
    L1 62.5% / L2 45.5%; + ladder 61.8 / 42.8; + ladder + cp_loss 0.02
    58.8 / 48.5. At 5% share, self-play and the ladder are not measurable
    against the j4 baselines (64.0-63.7 / 45.0-46.0). The weak pawn-space loss
    hints at better conversion on L2 (repetition 104 → 76,
    repetition-while-winning 65 → 43, wins 39 → 59) but L1 losses rose to 38,
    so it is inconclusive at 0.02; `sweeps/june4_selfplay_cp.json` tests 0.1.
13. **cp_loss 0.1: no clear conversion gain.** Shards 0-3 + self-play with
    the ladder: h256 L1 60.5% / L2 43.8%; h512 64.8 / 49.0 (h512 baseline
    63.7 / 49.8). Repetition still ~45% of games, repetition-while-winning
    59-61 (L1) and 47-51 (L2). None of WDL scale, mate ladder, progress
    shaping or the pawn-space loss fixed conversion. What measurably helped:
    more human data (+7 L1 from 726k → 2.9M) and self-play positions at an
    ~18% share (+5-7 L1 via fewer losses). Last test before the interruption:
    `sweeps/june4_selfplay5x.json` (self-play shard repeated 5× ≈ 21% share).
14. **Repeating the self-play shard 5× (~21% share) did not help.** h512 +
    ladder + cp_loss 0.1: L1 60.8% / L2 44.5% (1×: 64.8 / 49.0); h256-s400:
    64.0 / 48.0 (j4 baselines 64.0-63.7 / 45.0-46.0). Duplicated positions
    likely overfit; the shard-0 self-play gain came from fresh positions.
15. **(2026-09-15, after the reboot) July shards 0-1 add nothing.** h512 on
    June 0-3 + July 0-1 (4.36M): L1 65.2% / L2 50.0%, repetition 89 / 102,
    repetition-while-winning 59 / 59 — the same as June alone (63.7 / 49.8).
    h256: 62.3 / 49.2 (June alone 64.0 / 45.0), within seed noise.
    More human-game positions at depth 9 have plateaued; local labeling was
    stopped after July shard 1 to free the CPU.
16. **Lichess eval database as a data source.** `lichess_db_eval.jsonl.zst`
    (22 GB) holds positions with deep Stockfish analyses and best-move lines,
    so no local labeling is needed. `chess_nn.import_evals` filters them like
    `label` (not in check, quiet legal best move, no official openings). On a
    200k-line sample: 72.7% kept, median depth 30, 18.6% mate scores (human
    data 2.0%), 24% of positions with ≤10 pieces — the won endgames the net
    fails to convert.
17. **Eval-database pilot hurts.** First 10M lines at 40% sampling → 2.80M
    positions (61 s). h512 on those alone: L1 52.0% / L2 33.2% (losses
    60 / 101). Mixed with June 0-3 + July 0-1 (7.17M, 39% eval-db): L1 60.5% /
    L2 45.5%, repetition-while-winning 70 / 55 — below the human-only 65.2 /
    50.0 on both levels. Likely cause: analysis-board positions and depth-30+
    middlegame scores the depth-1 net cannot learn. Next test: only the
    eval-db endgames (≤10 pieces) mixed in at a small share
    (`sweeps/evpilot_eg.json`: 587,700 pilot positions, 50% mates, ~12% of
    the mix). Caveat: the file is not shuffled — 200k lines at line 60M have
    median depth 23 (start: 30), 10.3% mates (18.6%), 15% ≤10 pieces (24%) —
    so the pilot slice is not representative of a full-file sample.
18. **Eval-db endgames at ~12% share: no clear gain.** June 0-3 + July 0-1 +
    587,700 pilot positions with ≤10 pieces (h512): L1 64.8% / L2 49.2%
    (baseline 65.2 / 50.0); with the mate ladder 68.0 / 46.5. L1 +2.8 is
    inside seed noise and L2 fell; repetition-while-winning stays 59-65.
    Seed-1 replicate of the ladder run: 65.2 / 42.0, so the 68.0 was noise
    and L2 is worse on both seeds. Eval-db endgames are ruled out; the full
    eval-db conversion is dropped (the download is kept in `data/`).
    Loop-children generation (jj6-h512 vs L1, 8000 games): 2,280 loops
    (28.5%), 347,571 positions in 523 s.
19. **Loop children are slow to label.** `label --bench 2000` at depth 9 +
    deep 14 (≥600 cp): 103 positions/s with datagen sharing the CPU, 52.9%
    kept (most loop children are lopsided, so most get the deep re-search).
    Candidates from L1 + L2 are deduplicated, shuffled (seed 0) and labeled
    in 100k shards so partial results are usable and a restart loses at most
    one shard.
20. **Loop-children data (`data/shards_loops/shard_loops_a_*.npz`).** L2
    generation: 8000 games, 2,456 loops (30.7%), 359,188 positions. L1 + L2:
    706,614 unique candidates → 391,594 labeled positions (55% kept, ~140/s,
    ~80 min). 6.3% mates; non-mate |score| ≥ 300 cp 81.5%, ≥ 600 cp 57.0%,
    < 100 cp 8.5%; median 16 pieces, 20.9% with ≤10. In
    `sweeps/loops_a.json` (June 0-3 + July 0-1 + loops) they are 8.2% of
    positions — below the ~18% share at which fresh self-play helped (11).
19. **compete's W/D/L tally is wrong** (`0.5 as u32` counts draws as losses);
   scores and pass/fail are correct. `evaluate.py` recounts from points.

## State at the interruption

- June labeling (`data/candidates.txt`, 6.84M candidates → ~5M positions) was
  stopped deliberately after `shard_0003` (2.9M positions) to give self-play
  labeling the CPU; shards 4-6 remain and resume picks up at `shard_0004`.
- Self-play positions (`data/selfplay_L2_a.txt`, 212,568 FENs from
  m2a-h256-shard0 vs Level 2 with 10% random moves) are labeled:
  `data/shards_selfplay/shard_selfplay_a_0000.npz` (154,944 positions).
- Every sweep listed under "Resuming" finished; results are in `runs/` and
  committed.
- The July 2017 download finished (`data/lichess_2017-07.pgn.zst`); July
  sampling was running when work stopped and will be cut off by the update.
  Re-run it from scratch (it overwrites `data/candidates_2017-07.txt`), then
  label July.

## Next steps after the interruption

**Leaderboard #1 is not beaten yet.** Best dev results (200 games per level):
`j4sp-h512-s400-ladder-cp10` L1 64.8% / L2 49.0%, `j4-h512-32x32-s400`
63.7 / 49.8, `j4-h256-32x32-s400` 64.0 / 45.0. No model reached 70% on dev
L1, so no official run was made.

What moved the numbers: more human data (+7 L1, 726k → 2.9M) and fresh
self-play positions at ~18% share (+5-7 L1 via fewer losses). What did not:
WDL scale 800/1000, mate ladder, progress shaping, pawn-space loss 0.02/0.1,
repeating one self-play shard. The remaining leak is conversion: ~45% of games
end in threefold repetition, most with the net clearly winning.

1. **Finish the data.** Resume June shards 4-6, then July; train h256 and h512
   on the full ~5M (then ~10M). This is the one lever that helped reliably.
2. **Fresh self-play at scale, targeted at loops.** Run `datagen` with the best
   model at full CPU (no competing jobs) to produce ≥1M new positions, not
   repeats. Add a datagen mode with `--random-move-prob 0` that keeps positions
   only from games ending in repetition while the net is ahead, so labels land
   exactly on the loop positions; label with `--deep-depth 14`.
3. **Retrain on June (+ July) + fresh self-play at ~20% share** and compare
   `repetition_while_winning` against the j4 baselines with ≥2 seeds; single
   runs differ by ~±3 points.
4. **Revisit h128 once data reaches ~9M.** The leaderboard model is h128 at 9M;
   here h128 trailed h256 at 2.9M, which may be a data-size effect.
5. Only once dev L1 ≥ ~72%: run `--official` for Level 1 (then Level 2).

## Resuming after an interruption

Every step is resumable: labeling skips finished shards (a partial shard is
redone), sweeps skip finished runs and levels, curl resumes with `-C -`.

```bash
cd training

# June labeling (resumes at the first missing shard)
uv run python -m chess_nn.label --candidates data/candidates.txt --depth 9

# Self-play labeling (if data/shards_selfplay/shard_selfplay_a_0000.npz is missing)
uv run python -m chess_nn.label --candidates data/selfplay_L2_a.txt \
  --out-dir data/shards_selfplay --prefix shard_selfplay_a \
  --depth 9 --deep-depth 14 --deep-threshold-cp 600 --workers 6

# Sweeps (skip finished runs and levels)
uv run python -m chess_nn.sweep sweeps/shard0_seeds.json
uv run python -m chess_nn.sweep sweeps/shard0_selfplay.json
uv run python -m chess_nn.sweep sweeps/june4.json
uv run python -m chess_nn.sweep sweeps/june4b.json
uv run python -m chess_nn.sweep sweeps/june4_shaped.json
uv run python -m chess_nn.sweep sweeps/june4_selfplay.json
uv run python -m chess_nn.sweep sweeps/june4_selfplay_cp.json
uv run python -m chess_nn.sweep sweeps/june4_selfplay5x.json

# Table of every run
uv run python -m chess_nn.report

# Lichess eval database (2026-09-15): download (resumable), then convert.
(cd data && curl -C - -o lichess_db_eval.jsonl.zst.part \
  https://database.lichess.org/lichess_db_eval.jsonl.zst \
  && mv lichess_db_eval.jsonl.zst.part lichess_db_eval.jsonl.zst)
# Pilot: first 10M lines of the file (works on the partial download too)
mkfifo /tmp/pilot.fifo
(zstd -dc data/lichess_db_eval.jsonl.zst* | head -n 10000000 > /tmp/pilot.fifo &)
uv run python -m chess_nn.import_evals --evals /tmp/pilot.fifo \
  --out-dir data/shards_evaldb_pilot --sample-prob 0.4
uv run python -m chess_nn.sweep sweeps/evpilot_only.json
uv run python -m chess_nn.sweep sweeps/evpilot_mix.json
# Full: ~354M lines sampled at 3% → ~7.7M positions (seeded; skips finished shards)
uv run python -m chess_nn.import_evals --evals data/lichess_db_eval.jsonl.zst \
  --out-dir data/shards_evaldb --sample-prob 0.03
uv run python -m chess_nn.sweep sweeps/evdb_only.json
uv run python -m chess_nn.sweep sweeps/evdb_mix.json
uv run python -m chess_nn.sweep sweeps/evpilot_eg.json   # pilot endgames only

# Loop children (2026-09-15): children of repetition positions the net thought
# it was winning, from jj6-h512 vs Levels 1 and 2 (200 games → 72 loops, ~12k positions)
(cd datagen && cargo build --release)
for L in 1 2; do datagen/target/release/datagen runs/jj6-h512-s400/model.onnx \
  --out data/loops_L${L}_a.txt --games 8000 --threads 10 --level $L --random-move-prob 0 \
  --sample-prob 0 --loop-plies 16 --min-eval 3 --seed 1$L; done
# dedupe + shuffle (seed 0); rerunning this reproduces the same order, so label resumes
uv run python -c "import random; fens = list(dict.fromkeys(l.strip() for f in ['data/loops_L1_a.txt', 'data/loops_L2_a.txt'] for l in open(f) if l.strip())); random.Random(0).shuffle(fens); open('data/loops_a.txt', 'w').write('\n'.join(fens) + '\n')"
uv run python -m chess_nn.label --candidates data/loops_a.txt --out-dir data/shards_loops \
  --prefix shard_loops_a --depth 9 --deep-depth 14 --deep-threshold-cp 600 --shard-size 100000
uv run python -m chess_nn.sweep sweeps/loops_a.json

# July download, then sampling and labeling
(cd data && curl -C - -o lichess_2017-07.pgn.zst.part \
  https://database.lichess.org/standard/lichess_db_standard_rated_2017-07.pgn.zst \
  && mv lichess_2017-07.pgn.zst.part lichess_2017-07.pgn.zst)
uv run python -m chess_nn.sample_positions --pgn data/lichess_2017-07.pgn.zst \
  --out data/candidates_2017-07.txt --max-positions 7000000 --seed 3
uv run python -m chess_nn.label --candidates data/candidates_2017-07.txt \
  --prefix shard_2017-07 --depth 9
```
