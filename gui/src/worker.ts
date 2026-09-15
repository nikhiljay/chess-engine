// Runs the net's search off the page's main thread, so dragging and premoves
// stay smooth while it thinks.

import { createEngine } from "./engine.ts";
import type { AnalyseRequest, AnalyseResponse, WeightGlobals } from "./protocol.ts";

interface WorkerScope extends WeightGlobals {
  importScripts(...urls: string[]): void;
  postMessage(message: AnalyseResponse): void;
  onmessage: ((event: MessageEvent<AnalyseRequest>) => void) | null;
}

const scope = self as unknown as WorkerScope;
scope.importScripts("weights.js");
const engine = createEngine(scope.NET_META, scope.NET_WEIGHTS_B64);

scope.onmessage = (event) => {
  const { id, fen } = event.data;
  scope.postMessage({ id, result: engine.analyse(fen) });
};
