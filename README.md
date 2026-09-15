# chess-nn

Training pipeline for an ONNX eval network for the chess-challenge harness in
`..`. Design: `docs/specs/2026-09-14-nn-eval-training-design.md`.

Requires `uv`, `stockfish` (`brew install stockfish`), and the harness binaries
built with a current Rust (`cargo build --release -p cli` in `..`).

## Pipeline

```bash
# 1. Sample quiet-candidate positions from a Lichess month
uv run python -m chess_nn.sample_positions \
  --pgn data/lichess_2017-06.pgn.zst --out data/candidates.txt --max-positions 7000000

# 2. Dev openings (disjoint from the official 25)
uv run python -m chess_nn.dev_openings --pgn data/lichess_2017-06.pgn.zst

# 3. Label with Stockfish (benchmark first; resumable by shard)
uv run python -m chess_nn.label --candidates data/candidates.txt --bench 4000 --depth 9
uv run python -m chess_nn.label --candidates data/candidates.txt --depth 9

# 4. Train, export, validate (writes runs/<name>/)
uv run python -m chess_nn.train --name v1-h256 --epochs 30

# 5. Play dev openings; official openings only to confirm a milestone
uv run python -m chess_nn.evaluate runs/v1-h256/model.onnx --levels 1 2
uv run python -m chess_nn.evaluate runs/v1-h256/model.onnx --levels 1 --official
```

## Tests

```bash
uv run pytest -m "not slow"   # unit tests
uv run pytest -m slow          # end-to-end: label → train → export → compete
```
