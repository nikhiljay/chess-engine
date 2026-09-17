// Bundles the page and search worker into dist/ and generates dist/weights.js
// from data/weights.json (written by `uv run python -m chess_nn.export_gui`).
//
// Usage: npm run build

import { build } from "esbuild";
import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import type { NetMeta } from "../src/engine.ts";
import { SOUNDS, THEME } from "../src/sound-theme.ts";

const root = fileURLToPath(new URL("..", import.meta.url));
const dist = `${root}dist`;

await mkdir(dist, { recursive: true });
const common = { bundle: true, format: "iife", target: "es2020", logLevel: "warning" } as const;
await build({ ...common, entryPoints: [`${root}src/main.ts`], outfile: `${dist}/main.js` });
await build({ ...common, entryPoints: [`${root}src/worker.ts`], outfile: `${dist}/engine-worker.js` });
await copyFile(`${root}index.html`, `${dist}/index.html`);

// Only the active sound set ships; see sounds/README.md.
const sounds = [...new Set(Object.values(SOUNDS))];
await mkdir(`${dist}/sounds`, { recursive: true });
for (const file of sounds) await copyFile(`${root}sounds/${THEME}/${file}`, `${dist}/sounds/${file}`);

const weights = JSON.parse(await readFile(`${root}data/weights.json`, "utf8")) as { meta: NetMeta; weights_b64: string };
// `self` is the window on the page and the global scope in the worker.
const script = `self.NET_META = ${JSON.stringify(weights.meta)};\nself.NET_WEIGHTS_B64 = ${JSON.stringify(weights.weights_b64)};\n`;
await writeFile(`${dist}/weights.js`, script);

console.log(`built ${dist}: index.html, main.js, engine-worker.js, weights.js (${weights.meta.run}), sounds/ (${THEME}: ${sounds.join(", ")})`);
