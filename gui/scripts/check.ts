// Checks the TypeScript engine against published perft counts and against the
// harness: play-move choices and ONNX evals from data/reference.json.
//
// Usage: npm run check   (after `uv run python -m chess_nn.export_gui <run>`)

import { readFileSync } from "node:fs";
import { createEngine, type NetMeta } from "../src/engine.ts";

interface Reference {
  fen: string;
  harness_uci: string;
  onnx_eval: number;
}

const data = new URL("../data/", import.meta.url);
const weights = JSON.parse(readFileSync(new URL("weights.json", data), "utf8")) as { meta: NetMeta; weights_b64: string };
const reference = JSON.parse(readFileSync(new URL("reference.json", data), "utf8")) as Reference[];
const engine = createEngine(weights.meta, weights.weights_b64);
let failures = 0;

// Standard perft positions with published node counts.
const PERFT: [string, number, number][] = [
  ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197281],
  ["r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862],
  ["8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4, 43238],
  ["r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3, 9467],
  ["rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379],
];
for (const [fen, depth, expected] of PERFT) {
  const got = engine.perft(fen, depth);
  if (got !== expected) failures++;
  console.log(`perft ${depth} ${got === expected ? "ok  " : "FAIL"} ${got} (expected ${expected})  ${fen}`);
}

let agree = 0;
let maxDiff = 0;
const timings: { fen: string; ms: number; nodes: number }[] = [];
for (const ref of reference) {
  maxDiff = Math.max(maxDiff, Math.abs(engine.evaluate(ref.fen) - ref.onnx_eval));
  const result = engine.analyse(ref.fen);
  timings.push({ fen: ref.fen, ms: result.ms, nodes: result.nodes });
  if (result.best.uci === ref.harness_uci) {
    agree++;
    continue;
  }
  failures++;
  const harness = result.moves.find((m) => m.uci === ref.harness_uci);
  const harnessScore = harness ? `${harness.score.toFixed(5)}${harness.exact ? "" : " (bound)"}` : "not generated";
  console.log(`MISMATCH ${ref.fen}\n  engine ${result.best.uci} ${result.best.score.toFixed(5)} | harness ${ref.harness_uci} ${harnessScore}`);
}
if (maxDiff > 1e-4) failures++;

timings.sort((a, b) => b.ms - a.ms);
const mean = timings.reduce((sum, t) => sum + t.ms, 0) / timings.length;
console.log(`\nmoves agree ${agree}/${reference.length}; eval max |diff| ${maxDiff.toExponential(2)}`);
console.log(`analyse: median ${timings[timings.length >> 1].ms.toFixed(0)} ms, mean ${mean.toFixed(0)} ms, slowest ${timings[0].ms.toFixed(0)} ms (${timings[0].nodes} nodes)`);
process.exitCode = failures ? 1 : 0;
