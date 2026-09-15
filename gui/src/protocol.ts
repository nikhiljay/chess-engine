// Messages between the page and the search worker.

import type { Analysis, NetMeta } from "./engine.ts";

export interface AnalyseRequest {
  id: number;
  fen: string;
}

export interface AnalyseResponse {
  id: number;
  result: Analysis;
}

/** Globals set by the generated weights.js, in both the page and the worker. */
export interface WeightGlobals {
  NET_META: NetMeta;
  NET_WEIGHTS_B64: string;
}
