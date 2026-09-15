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
21. **Low-memory kills are not a training leak.** The loops_a sweep was
    killed twice by the harness's low-memory guard, both ~2 min into
    loopa-h512 (epoch 17-18). A direct run with 5 s memory sampling finished
    all 20 epochs: process footprint 3.3 GB while loading, then flat at
    2.7 GB; free memory 49-50%, with brief wired-memory swings to 8.6-9.9 GB
    (35-38% free) from other processes after the OS update. That run is
    reused as `runs/loopa-h512-s400` (same shards, order, seed and settings).
    Sweeps now run with `PYTHONUNBUFFERED=1` so logs survive a kill.
    A third kill hit during loopa-h256's games; at that moment the ChatGPT
    app's Codex renderer held 8.6 GB, the compressor ~11 GB, and swap was 0.
22. **Loop children at 8.2%: best L1 yet, but L2 falls and conversion is not
    fixed.** June 0-3 + July 0-1 + loops_a (h512): L1 69.0% / L2 43.8%
    (baseline 65.2 / 50.0), losses 19 / 68, repetition-while-winning
    54 / 54 (59 / 59). With the mate ladder: 64.0 / 48.5, repetition-while-
    winning 65 / 64. L1 +3.8 is at the edge of seed noise; L2 −6.2 is beyond
    it. A seed-1 replicate of the plain run checks the L1 number.
    h256 + loops_a: L1 68.5% / L2 46.5% (June+July h256 62.3 / 49.2), L1
    losses 30 → 19 — the same L1 pattern as h512 (losses 25 → 19), so the L1
    gain holds across two widths. L2 again does not improve: 122 draws,
    repetition-while-winning 56 → 80.
23. **Loop parents for a ranking diagnostic.** `datagen --emit parents`
    writes the positions the net moved from in loops instead of their
    children. loopa-h512 vs L1, 100 games: 30 loops → 180 unique parents
    (6 per loop), 6 in check, 32.5 legal moves on average. Next: on one
    parent set, compare each model's harness move with Stockfish's top moves
    to test whether the net picks draw-ish moves when a winning one exists.
24. **At loop positions the net does not blunder; it shuffles among moves
    Stockfish also rates as winning.** 180 parents from note 23, Stockfish
    depth 10 (best move, and `searchmoves` for the net's move); 143 have a
    best move ≥ +300 cp. jj6-h512: agrees with Stockfish's best 32.2%,
    median loss 15 cp, mean 86 cp (capped at 1000); in winning parents its
    move stays ≥ +300 in 96.5% and drops to ≤ +100 in 0.7%. loopa-h512:
    28.9%, 15 cp, 78 cp, 97.2%, 1.4%. So per-move Stockfish scores barely
    separate progress from shuffling, and a loss that ranks siblings by
    them has little to learn from. What breaks a loop is long-horizon
    progress (mate distance, pawn advance, trades), which depth-10 cp does
    not express. The sibling-ranking loss is not pursued.
25. **Loop data replicate: the L1 gain was mostly noise and L2 is worse.**
    loopa-h512 seed 1: L1 65.5% / L2 42.8% (seed 0: 69.0 / 43.8). Loop-data
    h512 mean L1 67.3 / L2 43.3 against the human-only 65.2 / 50.0; h256
    68.5 / 46.5 against 62.3 / 49.2. Loop children at 8.2% trade a small L1
    gain for a clear L2 loss, so they are not used going forward.
26. **WDL scale at full data (June 0-3 + July 0-1, h512).** s800: L1 61.5% /
    L2 46.5%. s800 + cp_loss 0.1: 64.5 / 50.7, seed 1 65.0 / 47.0. s1000 +
    cp_loss 0.1: 63.7 / **57.2** — best L2 so far (+7.2 over s400's 50.0),
    L2 losses 49 → 37, wins 49 → 66, repetition-while-winning 59 → 66.
    Single seed; the s800 seeds already differ by 3.7 on L2.
    `sweeps/jj6_scale2.json` runs a seed-1 replicate, s1500 + cp 0.1 and
    s1000 + cp 0.2.
27. **Large WDL scale + pawn loss is a real, modest gain.** s1000 + cp 0.1
    seed 1: L1 **72.0%** / L2 50.7% (seed 0: 63.7 / 57.2); s1500 + cp 0.1:
    69.0 / 53.0; s1000 + cp 0.2: 69.2 / 53.2 (L1 losses 14). Mean of the four
    runs L1 68.5 / L2 53.5 against s400's 65.2 / 50.0 — all four beat the
    baseline on at least one level and none falls below it on both. The 72.0
    is the best of four noisy runs, not the recipe's level, so it is not
    treated as the ≥72% official-run milestone. Next:
    `sweeps/jj6_scale3.json` (s1500 + cp 0.2, s1000 + cp 0.2 seed 1, h1024).
28. **s1000 + cp 0.2 puts dev L1 at ~70%; L2 stays in the low 50s.** s1000 +
    cp 0.2 seed 1: 70.8 / 49.2 (seed 0 69.2 / 53.2 → mean 70.0 / 51.2);
    s1000 + cp 0.1 seeds mean 67.9 / 54.0. s1500 + cp 0.2: 60.5 / 50.7, so
    both knobs high at once hurts. h1024 (856k params) + s1000 + cp 0.2:
    69.8 / 54.2 with 13 L1 losses — width barely moves L2. L2
    repetition-while-winning is still 67-78: converting those games is
    worth ~15-20 L2 points, so conversion remains the path to L2.
    Next: `sweeps/jj6_shape2.json` — progress shaping (finding 8) retried at
    3-7× the bonuses on top of s1000 + cp 0.2, where the pawn loss keeps
    gradient in won positions.
29. **Strong progress shaping hurts, even with the pawn loss.** On s1000 +
    cp 0.2 (unshaped seeds mean L1 70.0 / L2 51.2): mid bonuses (pawn 40,
    trade 15, mop-up 50 cp, threshold 300) 67.2 / 48.5 and 66.5 / 46.2
    (mean 66.9 / 47.4); strong (80 / 30 / 100) 60.2 / 42.5. Repetition-
    while-winning drops only slightly (L2 67-70 → 56-59) while losses rise
    (L2 41-51 → 57-74): the bonus makes the net overvalue pawn pushes and
    trades. Shaping is ruled out.
30. **Wider head and longer training: no gain.** On s1000 (h512): head
    128×32 (530k params) + cp 0.2: L1 69.2% / L2 51.7%; + cp 0.1: 68.5 /
    46.5; 40 epochs + cp 0.2: 64.2 / 49.2. The head and epoch count are not
    the limit.
31. **King-bucketed feature transformer (`king_buckets: true`).** Each
    perspective's transformer is chosen by its own king square: queenside /
    kingside × back rank / 2nd / 3rd-4th / 5th-8th = 8 buckets
    (`model.KingBucketNet`). Training stores shared + per-bucket delta
    weights so rare buckets start from the shared transformer; export merges
    them and picks each position's bucket with ArgMax → Gather inside the
    graph, so inference costs one transformer multiply. h512 ≈ 3.19M
    harness-counted params, h256 ≈ 1.6M. Tests: zero deltas equal EvalNet,
    ONNX parity, harness validation and param count
    (`tests/test_king_buckets.py`). First sweep: `sweeps/kb1.json`.
32. **King buckets: no overall gain.** s1000 on June 0-3 + July 0-1:
    kb-h512 + cp 0.2 L1 67.8% / L2 51.2% (L2 W/D/L 66/73/61);
    kb-h256 + cp 0.2 69.5 / 52.2 (65/79/56); kb-h512 + cp 0.1 66.5 / 50.0.
    Mean 67.9 / 51.1 against ~70 / 51-54 without buckets. Buckets convert
    more L2 wins (repetition-while-winning 41-51 in two runs) but lose more
    games (56-61 L2 losses) — the losses are tactical, which king-relative
    piece weights do not address.
33. **Goal is now Level 3 (2026-09-15).** Every candidate is graded on L1-L3
    (`sweeps/grade_l3.json` grades the best existing runs). The L3 bot
    searches depth 3 with TT/PVS/null-move over a full handcrafted eval
    (tapered PSTs, mobility, king-attack count, pawn shield, passed pawns,
    structure, bishop pair, rook files). compete logs keep only per-game
    results, so losing positions need our own game records.
    Data: self-play from jj6-h1024-s1000-cp20 against L2 and L3 (12k games
    each, 10% random moves, `data/selfplay_{L2,L3}_b.txt`), labeled first,
    then the remaining human shards.
34. **Level 3 baseline: ~29%.** jj6-h1024-s1000-cp20 58/200 (29%, 235 s);
    jj6-h512-s1000-cp10-seed1 56/200 (28%). Passing needs 140/200.
35. **Diagnosis with game history (user's plan, 2026-09-15).** Finding 24
    gave Stockfish loop positions without the game, so a shuffle back into
    a repeated position looked as good as progress. `datagen --record` now
    writes whole games; `chess_nn.diagnose` replays each with
    `position fen <start> moves ...` so Stockfish scores repetitions as
    draws, and reports repetition draws the net really had won, the move
    where the win slipped (and whether it repeated a position), and each
    loss's biggest drop. Games use random training openings; the dev
    openings stay for measurement. Next: a ranking objective on those
    decisions (good continuation vs the tempting move), then capacity.
    ThreatNet (in-graph attack/threat features, `threat_hidden`) is written
    and tested but parked until the ranking signal is measured.
36. **Recorded games (h1024, random training openings, no random moves).**
    L2: 400 games, score 46.6% — 103 wins, 167 draws (all threefold
    repetition), 130 losses (all checkmate), median 69 plies. L3: 24.2% —
    35 wins, 124 repetition draws, 241 checkmate losses, median 71 plies.
    No stalemates or fifty-move draws; games run to mate. Against L3 the
    main leak is losses (60% of games), not conversion, so the ranking pairs
    keep every clearly worse move (gap ≥ 100 cp), not only repetition slips.
    Also L3: jj6-h512-s1000-cp10 57.5/200 (29%).
37. **L3 on dev openings (W/D/L).** jj6-h1024-s1000-cp20 29.0% (20/76/104);
    jj6-h512-s1000-cp10-seed1 28.0% (23/66/111); jj6-h512-s1000-cp10 28.7%
    (16/83/101); kb-h256-s1000-cp20 33.2% (31/71/98); kb-h512-s1000-cp20
    26.8% (18/71/111). Every model loses about half its L3 games; king
    buckets h256 is the best L3 score so far but within ~±3 of the rest.
38. **Stockfish diagnosis with history (h1024, depth 14, games the net did
    not win).** Net move = Stockfish best in 38.6% (L2) / 36.4% (L3).
    - L2 (297 games): 167 repetition draws, 150 (90%) genuinely won
      (median best +9.2 pawns); 143 slipped. 130 losses: 106 contain one
      blunder (median drop 6.7 pawns from +0.3, ply 32); 67 had been winning.
    - L3 (365 games): 124 repetition draws, 92 (74%) won (median +8.1);
      241 losses: 198 with one blunder (median drop 6.4 pawns from +1.4,
      ply 42); 130 (54%) had been winning first.
    - The flagged slip sits a median 2 (L2) / 10 (L3) plies before the end:
      Stockfish counts a repetition as a draw only when it would complete
      the threefold (or repeats after its root), so earlier shuffles back
      into once-seen positions still score as winning, while the win was
      first reached ~34 plies earlier. Hence `rank_pairs` adds a rule: a
      winning-side move into an already-seen position, when Stockfish's
      best is not one, is a pair regardless of the gap.
    - h1024 L2 alone: 1,380 score pairs — too few; a larger diagnosis
      (1,500 games per level from h1024 and kb-h256, depth 12) is running.
    - rank_pairs_a (h1024 L2 + L3): 3,104 pairs — 3,057 by score gap, 47 by
      the repetition rule; 2,786 train / 318 val. Of 746 net moves that
      repeated a position: 323 were not in a winning position, 232 were
      already score pairs, 80 matched Stockfish's best, 64 had a child in
      check or game over, 47 became rule pairs. So ~279 pairs cover winning
      shuffles and the rest are blunders — consistent with L3's losses.
40. **Ranking objective, first try (rank_a, h512 s1000 cp 0.2, 2,786 train
    pairs): play gets clearly worse.** Control w0 (identical to
    jj6-h512-s1000-cp20): L1 69.2 / L2 53.2 / L3 29.0, held-out pair
    accuracy 0.52 → 0.36 over training. w0.5: 57.0 / 44.2 / 18.8, accuracy
    0.46. w2: 60.0 / 42.8 / 21.8, accuracy 0.48. The loss drops 9-13 points
    at every level and held-out accuracy stays below 0.5, so it memorizes
    the pairs without generalizing, and at w >= 0.5 it outweighs the score
    losses (train loss ~0.36 vs rank ~0.7·w). The control shows the base net
    prefers the worse child in 64% of these pairs — they are real blind
    spots. Most pairs are blunders whose refutation is a tactic the static
    features do not expose, so next: many more pairs (diagnosis_b), much
    smaller weights (0.05, 0.2), and ThreatNet's attack/threat inputs, with
    and without ranking (`sweeps/threat_a.json`, `sweeps/rank_b.json`).
41. **Diagnostic game (human vs jj6-h1024-s1000-cp20, 32...Nxe4 33. Rh8#).**
    The harness search follows only captures, promotions and castling (or
    all moves in check), so a quiet mating reply is never examined from a
    position not in check (`chess_nn.mate_in_one`).
    - Through play-move, all six existing nets (h1024, h512, kb-h256,
      rank-a w0/w0.5/w2) choose 32...Nxe4.
    - After 32. f3 Black has 30 moves: 5 (every knight move — the knight
      guards h7/h8) allow 33. Rh8#, unseen by the search; 25 avoid mate in
      one, but with the game history Stockfish depth 20 rates all 25 lost
      (best 32...Bxd5 −8.6 pawns; 10 are forced mates). Here avoiding the
      immediate mate would not have saved the game; the position was
      already −8.0.
    - Pipeline: diagnose scores the decision (best Bxd5 −827, Nxe4 mated) but
      build_pairs dropped it as an already-lost position (best ≤ −300), and
      would only ever have kept Stockfish's best as the good side. No safe
      reply gives check, so the in-check exclusion removed nothing here.
      Fix: a `mate_in_one` pair kind — when the net's move allows an unseen
      mate in one and safe moves exist, up to 4 safe alternatives (Stockfish's
      best first) are paired against it regardless of the lost filter, and
      reported separately from score-gap and repetition-rule pairs.
42. **Static pair accuracy by kind (rank_pairs_a_val, 318 pairs; built
    before the mate kind).** Tactical score gap ≥300 cp (n=86) / moderate
    100-299 (n=223) / repetition rule (n=9) / all:
    - rank-a-h512-w0 (plain, no ranking): 0.384 / 0.345 / 0.444 / 0.358
    - rank-a-h512-w0.5: 0.442 / 0.448 / 0.778 / 0.456
    - rank-a-h512-w2: 0.430 / 0.498 / 0.556 / 0.481
    - threat-a-h512-t64-w0 (ThreatNet, no ranking): 0.512 / 0.413 / 0.222 / 0.434
    ThreatNet without any ranking loss ranks tactical pairs best (0.51 vs
    0.38 plain), consistent with threat features exposing consequences the
    search misses, but n=86 puts that at ~2.4 standard errors and n=9
    repetition pairs says nothing. Firmer evidence has to come from the
    larger diagnosis and from move selection on the held-out mate
    benchmark.
43. **ThreatNet without ranking does not play better.** threat-a-h512-t64-w0
    (580k params, same data and recipe): L1 66.5% (133/200) / L2 49.5% (99)
    / L3 27.0% (54), against the plain control rank-a-h512-w0's 69.2 / 53.2
    / 29.0. Its better static tactical pair accuracy (finding 42) has not
    turned into match results; the mate-in-one benchmark will show whether
    it changes move selection at the search's blind spot.
44. **Larger diagnosis, h1024 half (2,437 games not won, depth 12)
    confirms 38 at four times the size.** 1,125 repetition draws, 926 (82%)
    genuinely won (median best +8.2 pawns), 839 slipped; 1,312 losses, 1,059
    (81%) with one blunder (median ply 38), 628 (48%) after having been
    winning. The net's move matched Stockfish's best in 37.7% of decisions.
45. **Tightening finding 44: most large drops do not decide the game.** A
    turning move takes a defensible position (Stockfish best with history
    >= -100 cp) to a losing one (net's move <= -300 cp) — `chess_nn.turning`.
    Lost games with at least one turning move: L2 71/130 (55%), L3 142/241
    (59%), larger h1024 set 796/1,312 (61%). Large drop but no turning move:
    57 / 96 / 477; never defensible: 13 / 26 / 111. Of all large drops in lost
    games (larger set), 3,180 of 4,099 (78%) were not turning moves — already
    lost, or still ahead afterwards. First turning move: median ply 34 / 43 /
    37, median drop 676 / 616 / 561 cp, and it allowed an unseen mate in one in
    7/71, 7/142, 59/796 (~5-10%). So "81% of losses hinge on one blunder"
    overstated it: ~55-61% of losses contain a real turning move, usually
    exactly one, and the search's mate-in-one blind spot explains only a small
    share of them. Training analysis now prioritizes turning moves (the
    `decisive` flag on pairs).
46. **Capacity experiment on the strongest established setup (user
    direction, 2026-09-15).** The ranking objective failed in rank_a and
    ThreatNet without ranking played worse, so capacity is tested on plain
    EvalNet with s1000 + cp 0.2 on June 0-3 + July 0-1 (4.36M), 20 epochs,
    batch 16384, lr 1e-3, seed 0, graded L1-L3 on dev_openings_100 — the
    recipe of the current default, jj6-h1024-s1000-cp20 (856,163 params:
    69.8 / 54.2 / 29.0; 8.5 s/epoch, 2.8 min to train; games 67 / 72 / 235 s
    with a free CPU). `sweeps/capacity_a.json` adds h3072 (2,566,243 params)
    and h8192 (6,841,443). The harness limits only parameters (10M); ONNX
    files are ~4 bytes per parameter. Promising sizes get a seed-1 repeat
    before any default change.
47. **The Python search in `chess_nn.search_check` matches the harness.** On
    the 55 GUI reference positions with jj6-h1024-s1000-cp20, its move
    choice equals play-move in 54; in the 55th
    (8/6R1/8/8/3p4/2pK1Q2/6p1/6k1 b) every legal move (c2, Kh2, Kh1) scores
    exactly −100,000 (mated), so only the tie-break order differs. Search
    accuracy and selected-move results from it can be trusted.
48. **kb-h256 diagnosis (2,432 games not won, depth 12) matches h1024.**
    981 repetition draws, 785 (80%) genuinely won; 1,451 losses, 840 (58%)
    with a turning move (median first at ply 39, drop 559 cp), 112 never
    defensible; 55 of 840 first turning moves (6.5%) allowed an unseen mate
    in one; 3,576 of 4,586 large drops (78%) were not turning moves. The
    failure pattern is not specific to one architecture.
49. **Held-out mate-in-one benchmark (`data/mate_bench.jsonl`).** 4,200
    fresh games (h1024, kb-h256, h512 vs L2 and L3, 700 each, seeds 902/903)
    → 767 positions where the net's move allowed an unseen mate in one while
    a safe move existed; 25 excluded as ranking-pair sources, 731 unique.
    Stockfish depth 16 rates the best safe move as saving the game (> −300
    cp) in only 109 (15%); in the other 622 avoiding the mate only delays a
    loss. After re-filtering against the completed diagnoses:
    `data/mate_bench_heldout.jsonl`. Allowance rates are reported for
    defensible and lost positions separately.
50. **rank_pairs_b (all four diagnoses).** 30,297 pairs — 3,385 mate_in_one,
    26,591 score, 321 repetition; 3,292 decisive (defensible → losing).
    Train 27,178 (2,966 decisive); val 3,119 (382 mate, 737 tactical score
    ≥300 cp, 1,958 moderate, 42 repetition; 326 decisive). Games that fed
    rank_pairs_a land in the same split (the split hashes the game key), so
    b_val is held out for every run so far. Static pair accuracy on b_val
    (mate / tactical / moderate / repetition / all):
    - jj6-h1024-s1000-cp20 (default): 0.223 / 0.303 / 0.279 / 0.119 / 0.275
      — lowest partly because most pairs come from its own games' mistakes
    - rank-a-h512-w0: 0.230 / 0.376 / 0.371 / 0.476 / 0.357
    - rank-a-h512-w0.5: 0.296 / 0.429 / 0.431 / 0.548 / 0.416
    - rank-a-h512-w2: 0.304 / 0.441 / 0.448 / 0.381 / 0.428
    - threat-a-t64-w0: 0.220 / 0.487 / 0.435 / 0.357 / 0.420
    - threat-a-t64-w0.1: 0.251 / 0.517 / 0.481 / 0.571 / 0.462
    ThreatNet's static tactical edge over the matched plain net (0.487 vs
    0.376, n=737, ~4 SE) is now robust, yet it plays worse (finding 43): an
    unresolved mismatch. `chess_nn.search_check` on a stratified val sample
    tests whether static order survives quiescence and changes the chosen move.
51. **Mate-in-one benchmark through play-move (730 held-out positions: 109
    defensible, 621 lost) — selection-biased, so compare only on positions
    from other models' games.** Each model allows the mate in 100% of the
    positions its own games produced (h1024 40/40 and 197/197; rank-a-h512-w0,
    identical to the h512 source model, 39/39 and 200/200). On positions from
    h1024 + kb256 games (70 defensible / 421 lost), unseen mates still
    allowed: rank-a-h512-w0 47 (67%) / 289 (69%); w0.5 41 (59%) / 267 (63%);
    w2 44 (63%) / 275 (65%); threat-a-t64-w0 35 (50%) / 283 (67%);
    threat-a-t64-w0.1 39 (56%) / 253 (60%). On kb256-source positions alone
    (30 / 224), h1024 allows 19 (63%) / 133 (59%). No model avoids these
    reliably; ThreatNet w0's lower defensible rate is ~2 SE at n=70, and
    lost-position rates are indistinguishable. Treated as a small, secondary
    signal (5-10% of decisive errors).
52. **Static rankings partly fail to survive the search (search_check on a
    stratified b_val sample: 1,068 pairs incl. all 326 decisive and 42
    repetition).** static accuracy → accuracy after quiescence; play-move
    chose better / worse / other:
    - jj6-h1024-s1000-cp20: 0.291 → 0.185; 9% / 68% / 23% (its own games'
      mistakes dominate the sample)
    - rank-a-h512-w0: 0.365 → 0.301; 15% / 38% / 47% (decisive 14% / 37% / 49%)
    - rank-a-h512-w2: 0.424 → 0.364; 16% / 27% / 57% (decisive 16% / 25% / 59%)
    Rankings right statically but wrong after quiescence outnumber the
    reverse ~2.5:1 (109 vs 40-45), because captures in the search replace the
    child position with a quiescence leaf. Ranking cut "chose the worse
    move" from 38% to 27% (decisive 37% → 25%) but "chose Stockfish's
    better move" stayed ~15% — it moved choices to other moves, whose
    quality is not yet measured. Mate-in-one pairs survive least (search
    accuracy 0.03-0.14): the bad move is usually a capture that quiescence
    scores as winning material. ThreatNet results pending.
53. **ThreatNet with ranking (threat-a-h512-t64-w0.1, 629,422 harness
    params incl. ~49k fixed attack matrices): L1 62.7% (76/99/25) / L2
    51.5% (50/106/44) / L3 27.0% (17/74/109).** ThreatNet w0: 66.5 / 49.5 /
    27.0; matched plain control rank-a-h512-w0: 69.2 / 53.2 / 29.0; default
    jj6-h1024-s1000-cp20: 69.8 / 54.2 / 29.0. No ranking or ThreatNet model
    beats the default or the plain control in games.
54. **ThreatNet's static edge does change selected moves at error-prone
    decisions, yet it plays worse — mismatch still open.** search_check on
    the same 1,068-pair sample (static → search; chose better / worse /
    other; decisive better / worse):
    - rank-a-h512-w0: 0.365 → 0.301; 15% / 38% / 47%; decisive 14% / 37%
    - rank-a-h512-w2: 0.424 → 0.364; 16% / 27% / 57%; decisive 16% / 25%
    - threat-a-t64-w0: 0.428 → 0.385; 20% / 31% / 49%; decisive 20% / 30%
    - threat-a-t64-w0.1: 0.474 → 0.414; 20% / 27% / 54%; decisive 19% / 26%
    ThreatNet's rankings survive quiescence better (84-96 static-right /
    search-wrong vs 109) and pick Stockfish's better move more often (20% vs
    15%), but games favour the plain net (66.5/49.5/27.0 vs 69.2/53.2/29.0).
    The sample holds only positions where h1024 or kb-h256 erred, so it
    cannot show whether ThreatNet is worse at ordinary moves. Next: move
    quality on a random sample of all decisions from held-out games
    (`chess_nn.move_quality`: play-move choice scored by Stockfish with the
    game history — mean loss, agreement, turning-move rate).
55. **Held-out baseline for the default (jj6-h1024-s1000-cp20).** 1,400
    fresh games (700 each vs L2 and L3, datagen seeds 902/903, 4 threads,
    random training openings), diagnosed at depth 12: 1,132 not won, 41,356
    decisions, agreement with Stockfish's best 36.7%. 507 repetition draws,
    417 (82%) genuinely won. 625 losses: 363 (58%) with a turning move
    (median ply 38, drop 557 cp; 27 allowed an unseen mate in one), 246 with
    only non-turning large drops, 47 never defensible. A candidate must
    lower the turning-move count on games recorded with the same datagen
    flags. `chess_nn.move_quality` scores five runs' choices on 1,500
    sampled decisions from these games.
56. **Capacity h3072 (2,566,243 params) does not beat the default.** L1
    67.0% (134/200) / L2 54.8% (109.5) / L3 25.0% (50) against h1024's 69.8
    / 54.2 / 29.0; val loss 0.00273 vs 0.00287. Cost: ~33 s/epoch (~11 min
    for 20 epochs) and 453-514 s per level of games, with the CPU shared.
    Move quality on 1,500 random held-out decisions (agreement / mean loss /
    large-drop rate / turning moves among 981 defensible):
    h1024 0.358 / 86 cp / 7.5% / 29 (3.0%); rank-a-h512-w0 0.364 / 81 cp /
    7.0% / 31 (3.2%); rank-a-h512-w2 0.343 / 90 cp / 7.8% / 29 (3.0%) — no
    measurable difference at ~30 turning moves each. h8192 training.
57. **Move quality on 1,500 random held-out decisions, all five runs — no
    reliable differences.** Agreement / mean loss / large-drop rate / turning
    moves among 981 defensible: h1024 0.358 / 86 cp / 7.5% / 29;
    rank-a-h512-w0 0.364 / 81 / 7.0% / 31; rank-a-h512-w2 0.343 / 90 / 7.8% /
    29; threat-a-t64-w0 0.377 / 75 / 5.9% / 27; threat-a-t64-w0.1 0.362 / 83 /
    7.1% / 31. ThreatNet w0 is best on every column, but by ~1 SE (6 cp
    mean loss, 16 fewer large drops in 1,500, 4 fewer turning moves); its
    2-4 point game deficit is likewise inside seed noise (same-recipe seeds
    have differed by up to 8 L1 points: 63.7 vs 72.0). The ThreatNet
    "mismatch" is most likely noise on both sides; neither games nor move
    quality show a real difference. Ranking at w2 raised mean loss (81 → 90
    cp) — with finding 52, the objective as trained shifts choices without
    improving them. Smallest remaining ranking test (`sweeps/rank_b.json`):
    ~24k score pairs (mate-in-one and repetition off), weight 0.05, on the
    default h1024 recipe, seeds 0 and 1, with a matched seed-1 control.
58. **Capacity does not improve play on this recipe and data.** Matched
    data (4.36M), objectives (s1000 + cp 0.2), 20 epochs, seed 0, dev
    openings:
    - h1024, 856,161 params, 3.4 MB: val 0.00287; 2.8 min to train; L1 69.8
      (92/95/13) / L2 54.2 (60/97/43) / L3 29.0 (20/76/104)
    - h3072, 2,566,243 params: val 0.00273; 10.8 min; 67.0 (90/88/22) /
      54.8 (61/97/42) / 25.0 (16/68/116); games 453-514 s per level
    - h8192, 6,841,443 params, 27.4 MB: val 0.00262 (still falling: e10
      0.00275); 24.9 min; 72.0 (105/78/17) / 51.5 (58/90/52) / 26.8
      (14/79/107); games 633-788 s per level (shared CPU)
    Validation loss falls steadily with width but match scores do not
    follow: h8192's +2.2 L1 comes with −2.7 L2 and −2.2 L3, all inside seed
    noise, so no size earns a seed-1 repeat and h1024 stays the default.
59. **Comparison table and default decision (2026-09-15 13:19).** Matched
    dev openings; inference timed on a quiet CPU (ONNX Runtime, one thread;
    play-move includes model load). Move quality from finding 57; mate
    counts against h1024 and rank-a-h512-w0 are biased by benchmark
    selection (finding 51). ThreatNet w0.1's 28.7 training minutes include
    CPU contention.

| run | params | train min | L1 | L2 | L3 | move quality: agree · loss · turning | unseen mate allowed: defensible · lost | held-out pairs (static by kind; after search) | ms/pos ORT b1 / b32 · play-move ms |
|---|---|---|---|---|---|---|---|---|---|
| jj6-h1024-s1000-cp20 | 856,161 | 2.8 | 69.8% (92/95/13) | 54.2% (60/97/43) | 29.0% (20/76/104) | 0.358 · 86 cp · 29/981 | 90/109 · 468/621 | mate 0.22 (n=382); tact 0.30 (n=737); mod 0.28 (n=1958); rep 0.12 (n=42); after search 0.18; picks better 9% / worse 68% | 0.128 / 0.037 · 33.3 |
| rank-a-h512-w0 | 428,643 | 2.8 | 69.2% (91/95/14) | 53.2% (54/105/41) | 29.0% (15/86/99) | 0.364 · 81 cp · 31/981 | 86/109 · 489/621 | mate 0.23 (n=382); tact 0.38 (n=737); mod 0.37 (n=1958); rep 0.48 (n=42); after search 0.30; picks better 15% / worse 38% | 0.068 / 0.018 · 30.4 |
| rank-a-h512-w05 | 428,643 | 4.8 | 57.0% (68/92/40) | 44.2% (50/77/73) | 18.8% (7/61/132) | — | 70/109 · 398/621 | mate 0.30 (n=382); tact 0.43 (n=737); mod 0.43 (n=1958); rep 0.55 (n=42) | 0.068 / 0.018 · 29.6 |
| rank-a-h512-w2 | 428,643 | 4.9 | 60.0% (77/86/37) | 42.8% (48/75/77) | 21.8% (17/53/130) | 0.343 · 90 cp · 29/981 | 76/109 · 408/621 | mate 0.30 (n=382); tact 0.44 (n=737); mod 0.45 (n=1958); rep 0.38 (n=42); after search 0.36; picks better 16% / worse 27% | 0.067 / 0.018 · 29.8 |
| threat-a-h512-t64-w0 | 629,422 | 7.3 | 66.5% (85/96/19) | 49.5% (55/88/57) | 27.0% (15/78/107) | 0.377 · 75 cp · 27/981 | 62/109 · 426/621 | mate 0.22 (n=382); tact 0.49 (n=737); mod 0.43 (n=1958); rep 0.36 (n=42); after search 0.39; picks better 20% / worse 31% | 0.338 / 0.059 · 106.9 |
| threat-a-h512-t64-w01 | 629,422 | 28.7 | 62.7% (76/99/25) | 51.5% (50/106/44) | 27.0% (17/74/109) | 0.362 · 83 cp · 31/981 | 63/109 · 384/621 | mate 0.25 (n=382); tact 0.52 (n=737); mod 0.48 (n=1958); rep 0.57 (n=42); after search 0.41; picks better 20% / worse 27% | 0.318 / 0.058 · 106.5 |
| kb-h256-s1000-cp20 | 1,596,583 | 4.5 | 69.5% (101/76/23) | 52.2% (65/79/56) | 33.2% (31/71/98) | — | — | — | 0.063 / 0.083 · 34.0 |
| cap-h3072-s1000-cp20 | 2,566,243 | 10.8 | 67.0% (90/88/22) | 54.8% (61/97/42) | 25.0% (16/68/116) | — | — | — | 0.399 / 0.106 · 44.2 |
| cap-h8192-s1000-cp20 | 6,841,443 | 24.9 | 72.0% (105/78/17) | 51.5% (58/90/52) | 26.8% (14/79/107) | — | — | — | 1.025 / 0.284 · 67.6 |

    **No candidate beats the default in matched games; jj6-h1024-s1000-cp20
    (856,161 params, 0.037 ms/position batched) stays the default.** Every
    difference is inside seed noise; ranking at w0.5-2 is clearly worse.
    Capacity lowers validation loss but not match scores at 4.36M positions,
    at ~9× training time and ~8× batched inference for 6.8M params. Running:
    bounded rank_b (h1024, score pairs at w0.05, seeds 0-1, matched control).
60. **rank_b fails on both seeds; the ranking loss is retired (2026-09-15).**
    h1024, s1000 + cp 0.2, score pairs only at w0.05, each seed against a
    control identical but for rank_weight 0 (w0 loads no pairs). Paired
    per-game point differences on dev openings (SE counts an opening's two
    games as one unit):
    - L1: seed 0 59.0 (75/86/39) vs 69.8 (92/95/13), −0.107 ± 0.032; seed 1
      59.0 (76/84/40) vs 72.8 (112/67/21), −0.138 ± 0.036; pooled −0.122 ±
      0.024, 95% CI [−0.169, −0.076], openings better/worse 47/102
    - L2: 40.2 (39/83/78) vs 54.2 (60/97/43), −0.140 ± 0.040; 46.0 (45/94/61)
      vs 56.5 (71/84/45), −0.105 ± 0.038; pooled −0.122 ± 0.028, CI [−0.176,
      −0.069], 44/98
    - L3: 23.2 (16/61/123) vs 29.0 (20/76/104), −0.058 ± 0.032; 20.0
      (13/54/133) vs 32.2 (23/83/94), −0.122 ± 0.032; pooled −0.090 ± 0.023,
      CI [−0.134, −0.046], 48/93
    Repetition draws fell (L1 86/84 vs 95/67) only because losses rose;
    repetitions the net was winning also fell (65/52 vs 78/49) while wins
    fell. No weight or kind variation follows; the failure analysis
    (which safe decisions became losing moves, and whether before or during
    quiescence) is finding 62.
61. **The ~99k baseline is much weaker than h1024 (2026-09-15).** EvalNet
    h128, no hidden head, WDL scale 800, no cp loss, same 4.36M positions
    and 20 epochs (the leaderboard's st-h128-s800-9M architecture as
    inferred from its name; unconfirmed). 98,947 params, 0.8/0.7 min to
    train (h1024: 2.8/3.2), ORT 0.020/0.005 ms per position at batch 1/32
    and 29 ms play-move (h1024 0.129/0.036, 37 ms; quiet CPU). Paired
    against the h1024 control of the same seed:
    - L1: 60.8 (81/81/38) vs 69.8 (92/95/13), −0.090 ± 0.033; 63.5
      (73/108/19) vs 72.8 (112/67/21), −0.092 ± 0.038; pooled −0.091 ±
      0.025, CI [−0.141, −0.042], openings better/worse 51/98
    - L2: 38.8 (34/87/79) vs 54.2 (60/97/43), −0.155 ± 0.034; 38.5
      (36/82/82) vs 56.5 (71/84/45), −0.180 ± 0.033; pooled −0.168 ± 0.024,
      CI [−0.214, −0.121], 42/108
    - L3: 25.2 (15/71/114) vs 29.0 (20/76/104), −0.037 ± 0.032; 22.2
      (18/53/129) vs 32.2 (23/83/94), −0.100 ± 0.034; pooled −0.069 ±
      0.023, CI [−0.114, −0.023], 54/94
    It changes architecture and objective together (no head, s800, cp 0),
    so it does not say which part costs strength; it does say the smaller
    recipe on our data cannot reach L2 (it scores 38-39%). The leaderboard
    model passing L2 must come from data, training or details its name does
    not show, not from this architecture on our data. h1024 stays the
    default; no follow-up on the small architecture without new evidence.
62. **Why rank_b lost games: compact failure analysis (2026-09-15).** Same
    1,500 held-out decisions (h1024 games, never in pair training), each
    model choosing through play-move, Stockfish d12 with history
    (`chess_nn.rank_failure`, `chess_nn.paired decisions`):
    - Turning moves did not rise overall: seed 0 introduced 25 and
      prevented 16 (sign test p 0.21), seed 1 introduced 14 and prevented 19
      (p 0.49), pooled 39 vs 35. Two plain h1024 seeds give 15 vs 13, change
      666/1,500 choices (44%) and differ by −7.6 ± 3.6 cp mean loss. rank_b
      changed 830 and 842 choices (55-56%) with +5.1 ± 4.1 and +12.9 ± 4.1
      cp mean loss. So the ranking loss perturbed ordinary choices a little
      more than a reseed does, but this fixed sample, drawn from the
      control's own games, shows nothing like the −0.12 points per game
      the games show: the losses come from positions the ranking model
      reaches in its own games. Matched decisions are a weak instrument;
      games decide.
    - Where the 39 new turning moves arose: 26 before quiescence (the
      losing move's value was the candidate's static eval of its quiet
      child), 11 on a capture line's leaf, 2 in check. 24 of the 26 are
      refuted by a quiet move or quiet check that no capture search sees
      (e.g. Rf5?? c6b4 in r1bk3r/1ppp2pp/2n5/pN1N4/P2P2pq/BQ2P1b1/B2K4/5R2
      w; h3?? Qh2+ in r6r/1pp2kpp/4p3/p3qp2/P1B3n1/1P2P3/R2P1PPP/1Q3RK1 w).
      The play-move replica matched 38/39 choices.
    - What changed: the controls preferred the safe move by a median
      1.4/1.1 pawns, the candidates the losing move by 0.85/0.87. Seed 0
      mostly underrated the safe move (16 of 25; median safe-move shift
      −1.9 pawns, up to −9), seed 1 split 7/7. Shifts of several pawns on
      positions the pairs never contained show the pairwise loss moving
      absolute evaluations it had no target for, while the WDL/cp targets
      alone anchor them.
    To avoid repeating it: the errors that decide games are static evals of
    quiet children refuted by quiet moves, so the signal has to reach those
    positions as absolute targets on the leaves search actually evaluates,
    not as orderings of children whose absolute values then drift; and
    any such change is judged by games from its own play. The targeted-data
    experiment (decision_data keeps exactly those leaves, labeled like
    broad data) follows from this; no ranking variant does.
63. **The leaderboard server plays 2 games per level from the start
    position (verified 2026-09-15).** From the public API
    (paradigm.xyz/puzzles/api/v1/chess/leaderboard, /submissions/{id},
    /puzzles/api/chess/submissions/{id}/replays, /replays/{id}), fetched
    directly:
    - `st-h128-s800-9M` (@drCathieSo_eth, 2026-08-10, 99,358 params) is #3,
      not #1: L1 2/2, L2 2/2, L3 0/2, `totalGames: 2` per level, every game
      `openingIndex: 0`, all six replays start from the standard start
      position. No weights, code or description are public; the name's
      reading stays a guess (99,358 is not a plain 1540→128→1 EvalNet).
    - #1 `onnx minify` (93 params) and #2 `salieri's revenge^6` (94 params)
      each played 8 games: 2/2 at L1-L4, all opening 0.
    - Our local L1-L3 baseline bots choose every one of the 238 baseline
      moves in st-h128's six server games (`datagen replay_check`), so the
      server's opponents match this checkout and games are deterministic.
    - Harness main (8e3a59d) equals the local clone (nn.rs identical). PR
      #20 (full-window root, no NN check extensions) was merged on
      2026-03-17 but the repo was re-initialized after it and current main
      has neither change; which NN search build the server runs is not
      verifiable from replays without the model. Open issue #24 reports a
      local 83% vs server 50% at L1 and `totalGames: 2`, unanswered.
    So a server "pass" means winning two deterministic start-position
    games, not ≥70% over 50 games; it is not comparable to our 200-game
    dev scores and says little about general strength. The small-model gap
    to explain is not "99k passes L2 over 50 games while 856k scores 54%".
64. **Server-equivalent start-position test for 11 existing models
    (2026-09-15; measurement only, not used to select).** `chess_nn.evaluate
    --openings data/startpos_openings.txt --levels 1 2 3 4` (a pass needs
    2/2):
    | run | params | L1 | L2 | L3 | L4 |
    |---|---|---|---|---|---|
    | jj6-h1024-s1000-cp20 (default) | 856,161 | 1 (W L) | 1 (W L) | 0.5 (D L) | 1 (D D) |
    | rank-b-h1024-w0-seed1 | 856,163 | 1 (D D) | 1.5 (W D) | 0.5 (D L) | 0 |
    | b99k-h128-none-s800 | 98,947 | 1 (D D) | 1.5 (W D) | 0 | 0.5 |
    | b99k-h128-none-s800-seed1 | 98,947 | 1 (D D) | 0 | 1 (W L) | 0.5 |
    | rank-a-h512-w0 | 428,643 | 1 (W L) | 0.5 | 0 | 0.5 |
    | kb-h256-s1000-cp20 | 1,596,583 | 1 (D D) | 1 (W L) | 0 | 0 |
    | threat-a-h512-t64-w0 | 629,422 | 1.5 (W D) | 1 (D D) | 0 | 0 |
    | cap-h3072-s1000-cp20 | 2,566,243 | **2 (W W)** | 1 (W L) | 0.5 | 0.5 |
    | cap-h8192-s1000-cp20 | 6,841,443 | 1.5 (W D) | 0.5 | 0.5 | 0 |
    | rank-b-h1024-w005 | 856,161 | 1 (W L) | 0 | 0.5 | 0.5 |
    | rank-b-h1024-w005-seed1 | 856,161 | 1 (W L) | 1 (D D) | 1 (W L) | 0.5 |
    88 games: 15 wins, 42 losses by checkmate, 31 repetition draws (17 of
    them after the net had been winning). Only cap-h3072 would "pass" a
    level (L1); none passes L2. Two games per level cannot rank these
    models — seeds of one recipe differ from 0 to 1.5 at L2 — but it shows
    what a server pass takes: two wins in a row, where a repetition draw is
    as fatal as a loss. st-h128 won all four of its L1-L2 games by mate.
65. **Distance-to-mate playout labels cut repetition but lose more games
    (conv_a, 2026-09-15).** Premise: h1024's recorded L2 games reach a
    Stockfish-winning position in 54% of all games and fail to win them
    (finding 38), and no label source so far rises as mate approaches.
    `chess_nn.playouts`: 13,376 clearly winning positions (best ≥ +500)
    from training-source games the net did not win, played out by
    Stockfish depth 8 with game history (12,809 end in mate, median 69
    plies); each quiet position labeled with Stockfish's score plus 15 cp
    per ply inside the last 100 plies before mate, clamp 2500. 747,088
    positions (14.6% of the mix; median |label| 1,613 cp; 4 min CPU;
    +17% optimizer steps). Same h1024 recipe, seed 0, against
    jj6-h1024-s1000-cp20:
    - L1 67.5 (96/78/26) vs 69.8 (92/95/13): −0.023 ± 0.032
    - L2 49.5 (63/72/65) vs 54.2 (60/97/43): −0.048 ± 0.037; repetition
      draws 71 vs 97, repetitions while winning 45 vs 70
    - L3 24.5 (16/66/118) vs 29.0 (20/76/104): −0.045 ± 0.031
    - start position (2 games per level): L1 1/2, L2 0.5/2, L3 0/2
    The first intervention to cut winning repetitions clearly (−25 at L2),
    but only 3 became wins and losses rose by 22. Progress shaping
    (finding 29) and king buckets (finding 32) showed the same trade. Not
    replicated or expanded. Stockfish depth 12 with history on both models'
    L2 games the net did not win: losses after a winning position 38 vs 20,
    losses from level or worse positions 27 vs 23; repetition draws with a
    win 60 vs 86, the slip now a median 20 plies before the end (control
    2.5). Both reach a winning position in 81-83% of all L2 games and win
    37-39% of those. So the extra losses come from positions the net was
    converting: when it stops shuffling it walks into tactics it cannot
    see. Conversion is gated by tactical safety, not by the lack of a
    progress signal alone.
66. **Leaves at winning decisions do not improve play (tact_a,
    2026-09-15).** Tests whether finding 65's losses from won positions can
    be trained out with ordinary labels. `decision_data --select winning
    --min-cp 300 --sample 20000`: for 20,000 sampled training-source
    decisions with best ≥ +300, every legal move's quiescence leaf (278,966
    children, 366,148 leaves, 3,186 terminal → 624,299 unique; 23 min),
    labeled Stockfish depth 9 like broad data → 547,128 positions (11.1% of
    the mix, +12.5% optimizer steps). Same h1024 recipe, seed 0, against
    jj6-h1024-s1000-cp20 (paired, per-opening SE):
    - L1 69.8 (105/69/26) vs 69.8 (92/95/13): 0.000 ± 0.033
    - L2 53.5 (62/90/48) vs 54.2 (60/97/43): −0.008 ± 0.033,
      CI [−0.073, +0.058]; repetition draws 90 vs 97, repetitions while
      winning 59 vs 70
    - L3 32.5 (21/88/91) vs 29.0 (20/76/104): +0.035 ± 0.030,
      CI [−0.023, +0.093]
    - Stockfish depth 12 with history on L2 games not won: losses after a
      winning position 26 vs 20 (conv_a 38); repetition draws with a win 76
      vs 86; slip a median 2 plies before the end (control 2.5).
    - start position (2 games per level, measurement only): L1 1.5, L2 0.5,
      L3 1 (control 1 / 1 / 0.5)
    No credible L2 gain and losses from won positions did not fall, so the
    tactical-safety reading of finding 65 is not supported by this data. L1
    shows the same trade as conv_a, shaping and king buckets in milder form:
    13 more wins, 13 more losses, 26 fewer draws, unchanged score. The L3
    difference is inside noise and is not pursued. No size-matched control
    or seed-1 replicate (the rule requires a positive L2 result first). Not
    promoted, not expanded. Across findings 29, 32, 65 and 66 the
    interventions move games between draws and decisive results without
    raising points.
67. **Teacher versus student: the search is not the limit, leaf accuracy is
    (2026-09-15).** `datagen/src/bin/teacher.rs` ports NnEvalBot's reference
    search line for line (depth 1 + quiescence, same constants, ordering and
    tie-breaks) and swaps only the evaluator, every arm in pawns from the
    side to move. Verified before use: 7,037/7,037 net moves reproduced
    across h1024's 200 recorded dev L2 games, and 8/8 freshly played games
    identical to compete's recordings. 20 dev openings x both colours at L2
    (40 games), paired against h1024's recorded games on the same openings
    (student 56.2%, 11/23/6):
    - d9 (Stockfish depth 9 under our training transform: clamp ±2500 cp,
      mate → ±25 pawns): 93.8% (35/5/0), +0.375 ± 0.067,
      CI [+0.244, +0.506], 14 openings better, 0 worse
    - raw (same search, unclamped, mate distance kept): 96.2% (37/3/0),
      +0.400 ± 0.061, CI [+0.28, +0.52], 16/0. Against d9: +0.025 ± 0.017,
      CI [−0.009, +0.059] — no evidence the clamp costs anything here.
    - static (Stockfish's static eval, negated for Black): 68.8% (17/21/2),
      +0.125 ± 0.064, CI [−0.001, +0.251], 11 better / 4 worse
    - repetition draws: student 23, static 21, d9 5, raw 3; median game 66
      plies (student, static) vs 54-56 (d9, raw)
    - decision agreement with d9 on the student's own 40 games: 581/1,393
      (41.7%); 40% where the teacher is winning
    Depth-1 + quiescence wins these games outright when the leaf values are
    accurate, so search depth is not the binding constraint, and neither is
    the target clamp at L2. Conversion follows leaf accuracy with no
    progress term, no history and no king buckets, which is why shaping
    (29), king buckets (32), conv_a (65) and tact_a (66) moved behaviour
    without adding points. The teacher is a diagnostic, not a candidate
    (~15 ms/position, not a net). Stockfish's own static evaluation gains
    far less than its searched scores, so what plays well here is the
    searched score — that says our student is far from reproducing the
    target it is already trained on, not that a static net cannot learn it.
    Scope: L2, 40 games, one opening subset, one deterministic baseline.
39. **Why the earlier conversion fixes failed, given 38.** Every one of
    them changed per-position targets from Stockfish searches that did not
    know the game: mate ladder, WDL scale, pawn-space loss (5, 12-13, 26),
    eval-db endgames (17-18) and loop-children data (22, 25) all labeled a
    shuffle's child with the same winning score as a progress move's child,
    so the gradient never separated them (finding 24's 15 cp median gap).
    Progress shaping (8, 29) added bonuses to won human positions, where
    loop positions share the same features, and distorted non-quiet
    evaluations enough to raise losses. King buckets (32) added
    king-relative capacity, but the losses are single tactical blunders
    from level or better positions (6-7 pawns at move 16-21), which
    king-relative weights do not address. What none of them supplied is a
    signal that compares the moves available in the same position — the
    ranking objective.
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

**Updated 2026-09-15 afternoon (findings 26-29). Leaderboard #1 is not
beaten yet.** Best recipe: h512 on June 0-3 + July 0-1 (4.36M) with WDL scale
1000 and `cp_loss_weight` 0.1-0.2 — seed means L1 70.0 / L2 51.2 (cp 0.2) and
67.9 / 54.0 (cp 0.1), against 65.2 / 50.0 at scale 400. Candidate 2 below
worked modestly; candidate 3 (strong shaping) failed; width (h1024) adds
little. L2 is the gap: ~70 of 200 L2 games are still repetition draws the
net thinks it is winning. Work is paused pending candidate 1.

Earlier state (midday): `jj6-h512-s400` (June 0-3 + July 0-1, 4.36M) L1
65.2% / L2 50.0%; `j4-h512-32x32-s400` 63.7 / 49.8. No official run has been
made.

Ruled out since the interruption (findings 15-25): more human data (July),
the Lichess eval database whole or endgames-only, and loop-children
self-play data (L1 +~2, L2 −6). Earlier: WDL scale 800/1000 at 726k, mate
ladder, progress shaping at 10-15 cp, pawn-space loss 0.02/0.1, repeated
self-play. Finding 24: at loop positions the net plays moves Stockfish also
rates as winning (median loss 15 cp), so the leak is lack of progress, not
blunders, and per-move Stockfish scores cannot teach progress.

Candidate next steps, most informative first:
1. **Calibrate dev vs official (needs the user's OK).** One `--official`
   run of `jj6-h512-s400` at L1 and L2 — a measurement, not tuning. #1
   passes L2 at 70% with 99k params while our 429k net scores 50% on dev
   L2; the gap is either the opening sets or a real technique gap, and the
   next steps differ.
2. **Scale + pawn loss at full data.** WDL scale 800-1000 was only tried at
   726k; retry at 4.36M with `cp_loss_weight` 0.1, ≥2 seeds. In won
   positions the scale-400 target is saturated, leaving eval differences
   between winning moves unconstrained.
3. **Strong progress shaping in won endgames.** Earlier bonuses (10-15 cp)
   sat inside eval error; retry at 50-100 cp per step for |score| ≥ 600 with
   the pawn loss so the gradient survives.
4. **h128 at larger data** (#1's likely recipe: h128, scale 800, 9M).

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
