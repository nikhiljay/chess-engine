// Paradigm Puzzle Chess Engine: play the eval net with chess.com-style input — drag or tap,
// premoves, right-click marks and arrows, move review with the arrow keys —
// and see how the net scores every legal move in whatever position is shown.

import { MATE, createEngine, type Analysis, type Engine } from "./engine.ts";
import type { AnalyseRequest, AnalyseResponse, WeightGlobals } from "./protocol.ts";

interface SavedState {
  pgn?: string;
  human?: ChessColor;
  flipped?: boolean;
  soundOn?: boolean;
}

interface HotApi {
  ready?: (start: (data: SavedState) => void) => void;
  data?: SavedState;
  snapshot?: (collect: () => SavedState) => void;
}

declare global {
  interface Window extends WeightGlobals {
    claude?: { hot?: HotApi };
    webkitAudioContext?: typeof AudioContext;
  }
}

interface Premove {
  from: string;
  to: string;
}

interface Drag {
  from: string;
  pointerId: number;
  wasSelected: boolean;
  moved: boolean;
  startX: number;
  startY: number;
  hover: string | null;
  el: HTMLDivElement;
  size: number;
}

type Analyser = (fen: string) => Promise<Analysis>;

const GLYPH: Record<ChessPieceType, string> = { k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };
const TEXT_STYLE = "︎"; // keep the chess glyphs from rendering as emoji
const NAMES: Record<ChessPieceType, string> = { k: "king", q: "queen", r: "rook", b: "bishop", n: "knight", p: "pawn" };
const FILES = "abcdefgh";
const SHOWN_MOVES = 9;
const MIN_REPLY_MS = 300; // gives premoves a window and lets the reply read as a move
const ANIM_MS = 170;
const REVIEW_DEBOUNCE_MS = 120; // holding an arrow key scores only where you stop
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function $<T extends HTMLElement = HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} is missing`);
  return el as T;
}

let engine: Engine;
let analyseForMove: Analyser; // the net's own moves
let analyseForPanel: Analyser; // the panel, on a separate worker so it never delays a reply
let game: Chess;
let human: ChessColor = "w";
let flipped = false;
let selected: string | null = null;
let targets: string[] = [];
let thinking = false;
let keys: string[] = [];
let premoves: Premove[] = [];
let generation = 0; // bumped by new game / take back so stale net replies are ignored
let drag: Drag | null = null;
let returning: string | null = null;
let rightStart: { square: string; pointerId: number } | null = null;
let pendingPromotion: ChessMove[] | null = null;
let viewPly: number | null = null; // null = the live position
let soundOn = true;
let audio: AudioContext | null = null;
let panelTimer: ReturnType<typeof setTimeout> | null = null;
const analyses = new Map<string, Analysis>(); // by FEN; the search is deterministic
const analysing = new Set<string>();
const marks = { squares: new Set<string>(), arrows: new Map<string, Premove>() };
const squares: HTMLButtonElement[] = [];
const buttonAt = new Map<string, HTMLButtonElement>();

const positionKey = (chess: Chess): string => chess.fen().split(" ").slice(0, 4).join(" ");
const fileIndex = (square: string): number => FILES.indexOf(square[0]);
const rankIndex = (square: string): number => Number(square[1]) - 1;
const uciOf = (move: ChessMove): string => move.from + move.to + (move.promotion ?? "");
const sideName = (color: ChessColor): string => (color === "w" ? "White" : "Black");

// ------------------------------------------------------------------ engine

function createAnalyser(): Analyser {
  const local = (fen: string): Promise<Analysis> =>
    new Promise((resolve) => setTimeout(() => resolve(engine.analyse(fen)), 0));
  interface Job {
    fen: string;
    resolve: (result: Analysis) => void;
    timer: ReturnType<typeof setTimeout>;
  }
  let worker: Worker | null = null;
  let seq = 0;
  const pending = new Map<number, Job>();

  // If the worker can't start or stalls, finish its jobs on the page instead.
  function abandon(): void {
    if (!worker) return;
    worker.terminate();
    worker = null;
    for (const job of pending.values()) {
      clearTimeout(job.timer);
      void local(job.fen).then(job.resolve);
    }
    pending.clear();
  }

  try {
    worker = new Worker("engine-worker.js");
    worker.onmessage = (event: MessageEvent<AnalyseResponse>) => {
      const job = pending.get(event.data.id);
      if (!job) return;
      clearTimeout(job.timer);
      pending.delete(event.data.id);
      job.resolve(event.data.result);
    };
    worker.onerror = abandon;
  } catch {
    worker = null;
  }

  return (fen) => {
    const active = worker;
    if (!active) return local(fen);
    return new Promise((resolve) => {
      const id = ++seq;
      pending.set(id, { fen, resolve, timer: setTimeout(abandon, 15000) });
      const request: AnalyseRequest = { id, fen };
      active.postMessage(request);
    });
  };
}

// The engine speaks UCI; chess.js supplies the move names people read.
function withSan(result: Analysis, fen: string): Analysis {
  const names = new Map(new Chess(fen).moves({ verbose: true }).map((m) => [uciOf(m), m.san]));
  for (const m of result.moves) m.san = names.get(m.uci) ?? m.uci;
  result.best = result.moves.find((m) => m.uci === result.best.uci) ?? result.best;
  return result;
}

function fmtPawns(score: number): string {
  if (score >= MATE) return "mate";
  if (score <= -MATE) return "mated";
  return (score > 0 ? "+" : "") + score.toFixed(2);
}

// ------------------------------------------------------------------- rules

function rebuildKeys(): void {
  const replay = new Chess();
  keys = [positionKey(replay)];
  for (const m of game.history({ verbose: true })) {
    replay.move(m);
    keys.push(positionKey(replay));
  }
}

function positionAtPly(ply: number): Chess {
  const replay = new Chess();
  for (const m of game.history({ verbose: true }).slice(0, ply)) replay.move(m);
  return replay;
}

const shownPosition = (): Chess => (viewPly === null ? game : positionAtPly(viewPly));

interface Outcome {
  title: string;
  sub: string;
}

function outcome(): Outcome | null {
  if (game.in_checkmate()) {
    const winner: ChessColor = game.turn() === "w" ? "b" : "w";
    return winner === human
      ? { title: "Checkmate — you win", sub: "The net ran out of moves. Start a new game or take back to try another line." }
      : { title: "Checkmate — the net wins", sub: "Take back a move, or use ← to replay how it happened." };
  }
  if (game.in_stalemate()) return { title: "Stalemate — draw", sub: "The side to move has no legal move and is not in check." };
  const current = positionKey(game);
  if (keys.filter((k) => k === current).length >= 3) {
    return { title: "Threefold repetition — draw", sub: "This is the net's known leak: it shuffles in positions it thinks are winning." };
  }
  if (parseInt(game.fen().split(" ")[4], 10) >= 100) return { title: "Fifty-move rule — draw", sub: "Fifty moves each without a capture or pawn move." };
  if (game.insufficient_material()) return { title: "Insufficient material — draw", sub: "Neither side has enough pieces left to checkmate." };
  return null;
}

const yourTurn = (): boolean => !outcome() && !thinking && !pendingPromotion && game.turn() === human;
const premoveTime = (): boolean => !outcome() && game.turn() !== human;

function piecesOf(chess: Chess): Map<string, ChessPiece> {
  const pieces = new Map<string, ChessPiece>();
  for (const file of FILES) {
    for (let rank = 1; rank <= 8; rank++) {
      const piece = chess.get(file + rank);
      if (piece) pieces.set(file + rank, piece);
    }
  }
  return pieces;
}

// Pieces as you see them live: the real position with queued premoves applied.
function displayPieces(): Map<string, ChessPiece> {
  const pieces = piecesOf(game);
  for (const { from, to } of premoves) {
    const piece = pieces.get(from);
    if (!piece) continue;
    pieces.delete(from);
    if (piece.type === "k" && Math.abs(fileIndex(to) - fileIndex(from)) === 2) {
      const rank = from[1];
      const kingside = to[0] === "g";
      const rookFrom = (kingside ? "h" : "a") + rank;
      const rook = pieces.get(rookFrom);
      if (rook) {
        pieces.delete(rookFrom);
        pieces.set((kingside ? "f" : "d") + rank, rook);
      }
    }
    const promotes = piece.type === "p" && (to[1] === "8" || to[1] === "1");
    pieces.set(to, promotes ? { type: "q", color: piece.color } : piece);
  }
  return pieces;
}

// Premove destinations follow how the piece moves, ignoring what is in the way.
function premoveTargets(pieces: Map<string, ChessPiece>, from: string): string[] {
  const piece = pieces.get(from);
  if (!piece) return [];
  const fx = fileIndex(from);
  const fy = rankIndex(from);
  const castling = game.fen().split(" ")[2];
  const [kingsideRight, queensideRight] = piece.color === "w" ? ["K", "Q"] : ["k", "q"];
  const out: string[] = [];
  for (let x = 0; x < 8; x++) {
    for (let y = 0; y < 8; y++) {
      if (x === fx && y === fy) continue;
      const dx = x - fx;
      const dy = y - fy;
      const ax = Math.abs(dx);
      const ay = Math.abs(dy);
      let ok = false;
      switch (piece.type) {
        case "p": {
          const dir = piece.color === "w" ? 1 : -1;
          const start = piece.color === "w" ? 1 : 6;
          ok = (dx === 0 && (dy === dir || (dy === 2 * dir && fy === start))) || (ax === 1 && dy === dir);
          break;
        }
        case "n":
          ok = (ax === 1 && ay === 2) || (ax === 2 && ay === 1);
          break;
        case "b":
          ok = ax === ay;
          break;
        case "r":
          ok = dx === 0 || dy === 0;
          break;
        case "q":
          ok = ax === ay || dx === 0 || dy === 0;
          break;
        case "k": {
          const back = piece.color === "w" ? 0 : 7;
          const castle =
            fy === back && fx === 4 && dy === 0 &&
            ((x === 6 && castling.includes(kingsideRight)) || (x === 2 && castling.includes(queensideRight)));
          ok = (ax <= 1 && ay <= 1) || castle;
          break;
        }
      }
      if (ok) out.push(FILES[x] + (y + 1));
    }
  }
  return out;
}

// ------------------------------------------------------------------ render

const PIECE_TYPES: ChessPieceType[] = ["k", "q", "r", "b", "n", "p"];

// Chess glyphs sit off-centre in their em box (side bearings, baseline), so
// measure each glyph's ink in the loaded font and shift it to the true centre.
async function centerPieceGlyphs(): Promise<void> {
  const family = getComputedStyle(document.documentElement).getPropertyValue("--glyph").trim();
  const size = 200;
  const font = `${size}px ${family}`;
  try {
    await document.fonts.load(font, PIECE_TYPES.map((type) => GLYPH[type]).join(""));
  } catch {
    // Measure whichever font the browser falls back to.
  }
  const ctx = document.createElement("canvas").getContext("2d");
  if (!ctx) return;
  ctx.font = font;
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  const rules: string[] = [];
  for (const type of PIECE_TYPES) {
    const m = ctx.measureText(GLYPH[type] + TEXT_STYLE);
    // With line-height 1 the glyph box is one em tall and the baseline sits in the half-leading model's place.
    const baseline = (size - (m.fontBoundingBoxAscent + m.fontBoundingBoxDescent)) / 2 + m.fontBoundingBoxAscent;
    const inkCenterX = (m.actualBoundingBoxRight - m.actualBoundingBoxLeft) / 2;
    const inkCenterY = baseline + (m.actualBoundingBoxDescent - m.actualBoundingBoxAscent) / 2;
    const dx = (m.width / 2 - inkCenterX) / size;
    const dy = (size / 2 - inkCenterY) / size;
    if (Number.isFinite(dx) && Number.isFinite(dy)) {
      rules.push(`.piece[data-type="${type}"] { translate: ${dx.toFixed(4)}em ${dy.toFixed(4)}em; }`);
    }
  }
  let style = document.getElementById("piece-centering");
  if (!style) {
    style = document.createElement("style");
    style.id = "piece-centering";
    document.head.appendChild(style);
  }
  style.textContent = rules.join("\n");
}

function buildBoard(): void {
  const board = $("board");
  const marksLayer = $("marks");
  for (let i = 0; i < 64; i++) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sq";
    // Mouse and touch go through the board's pointer handlers; this handles keyboard activation.
    button.addEventListener("click", (event) => {
      if (event.detail === 0 && button.dataset.square) keyboardSquare(button.dataset.square);
    });
    board.insertBefore(button, marksLayer);
    squares.push(button);
  }
}

function findKing(chess: Chess, color: ChessColor): string | null {
  for (const [square, piece] of piecesOf(chess)) {
    if (piece.type === "k" && piece.color === color) return square;
  }
  return null;
}

function renderBoard(view: Chess): void {
  const live = viewPly === null;
  const pieces = live ? displayPieces() : piecesOf(view);
  const history = game.history({ verbose: true });
  const shownPly = viewPly ?? history.length;
  const last = shownPly > 0 ? history[shownPly - 1] : null;
  const checkSquare = view.in_check() && !(live && premoves.length) ? findKing(view, view.turn()) : null;
  const premoveSquares = new Set(live ? premoves.flatMap((p) => [p.from, p.to]) : []);
  const hidden = drag ? drag.from : returning;
  buttonAt.clear();
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const fileIdx = flipped ? 7 - col : col;
      const rank = flipped ? row + 1 : 8 - row;
      const name = FILES[fileIdx] + rank;
      const button = squares[row * 8 + col];
      const piece = pieces.get(name);
      button.dataset.square = name;
      buttonAt.set(name, button);
      const classes = ["sq"];
      if ((fileIdx + rank - 1) % 2 === 1) classes.push("light");
      if (last && (last.from === name || last.to === name)) classes.push("last");
      if (premoveSquares.has(name)) classes.push("premove");
      if (marks.squares.has(name)) classes.push("marked");
      if (selected === name) classes.push("selected");
      if (targets.includes(name)) classes.push("target", piece ? "capture" : "empty");
      if (checkSquare === name) classes.push("check");
      if (drag && drag.hover === name) classes.push("hover");
      button.className = classes.join(" ");
      let html = "";
      if (piece) {
        html += `<span class="piece ${piece.color}${live && hidden === name ? " lifted" : ""}" data-type="${piece.type}">${GLYPH[piece.type]}${TEXT_STYLE}</span>`;
      }
      if (col === 0) html += `<span class="coord rank">${rank}</span>`;
      if (row === 7) html += `<span class="coord file">${FILES[fileIdx]}</span>`;
      button.innerHTML = html;
      const pieceLabel = piece ? `, ${piece.color === "w" ? "white" : "black"} ${NAMES[piece.type]}` : "";
      button.setAttribute("aria-label", name + pieceLabel + (targets.includes(name) ? ", legal move" : ""));
    }
  }
  renderMarks();
}

function squareCenter(square: string): [number, number] {
  const col = flipped ? 7 - fileIndex(square) : fileIndex(square);
  const row = flipped ? rankIndex(square) : 7 - rankIndex(square);
  return [col * 100 + 50, row * 100 + 50];
}

function renderMarks(): void {
  let svg = "";
  for (const { from, to } of marks.arrows.values()) {
    const [x1, y1] = squareCenter(from);
    const [x2, y2] = squareCenter(to);
    const length = Math.hypot(x2 - x1, y2 - y1);
    const ux = (x2 - x1) / length;
    const uy = (y2 - y1) / length;
    const headLength = 40;
    const halfWidth = 27;
    const bx = x2 - ux * headLength;
    const by = y2 - uy * headLength;
    const f = (n: number): string => n.toFixed(1);
    svg += `<line class="arrow-shaft" x1="${f(x1 + ux * 24)}" y1="${f(y1 + uy * 24)}" x2="${f(bx)}" y2="${f(by)}"></line>`;
    svg += `<polygon class="arrow-head" points="${f(x2 - ux * 4)},${f(y2 - uy * 4)} ${f(bx - uy * halfWidth)},${f(by + ux * halfWidth)} ${f(bx + uy * halfWidth)},${f(by - ux * halfWidth)}"></polygon>`;
  }
  $("marks").innerHTML = svg;
}

/** "12. Rxh3" or "12… Nf6" for the move that reached `ply`. */
function moveLabel(ply: number): string {
  const san = game.history()[ply - 1];
  return `${Math.ceil(ply / 2)}${ply % 2 === 1 ? "." : "…"} ${san}`;
}

function renderStatus(view: Chess): void {
  const live = viewPly === null;
  const over = live ? outcome() : null;
  const evalStm = engine.evaluate(view.fen());
  const white = view.turn() === "w" ? evalStm : -evalStm;
  let readout = fmtPawns(white);
  let whiteShare = 1 / (1 + Math.exp(-white / 10));
  if (view.in_checkmate()) {
    readout = view.turn() === "w" ? "0–1" : "1–0";
    whiteShare = view.turn() === "w" ? 0 : 1;
  } else if (over) {
    readout = "½–½";
    whiteShare = 0.5;
  }
  $("readout").textContent = readout;
  $("gauge-fill").style.height = `${(whiteShare * 100).toFixed(1)}%`;
  $("gauge").classList.toggle("flipped", flipped);

  const title = $("status-title");
  const sub = $("status-sub");
  sub.classList.remove("warn");
  if (!live) {
    title.textContent = viewPly ? `After ${moveLabel(viewPly)}` : "Starting position";
    sub.textContent = `Move review. ← → step through the game; End or a tap on the board returns to it.${thinking ? " The net is thinking." : ""}`;
    return;
  }
  if (over) {
    title.textContent = over.title;
    sub.textContent = over.sub;
    return;
  }
  if (game.turn() !== human) {
    title.textContent = "The net is thinking…";
    if (premoves.length) {
      const [first] = premoves;
      const more = premoves.length > 1 ? `, then ${premoves.length - 1} more` : "";
      sub.textContent = `Premove queued: ${first.from}–${first.to}${more}. Right-click or Esc to cancel.`;
    } else {
      sub.textContent = "Drag a piece now to premove.";
    }
    return;
  }
  if (pendingPromotion) {
    title.textContent = "Choose a piece";
    sub.textContent = "Pick what your pawn becomes, or press Esc to put it back.";
    return;
  }
  title.textContent = game.in_check() ? "Check — your move" : "Your move";
  const seen = keys.filter((k) => k === positionKey(game)).length;
  if (seen === 2) {
    sub.textContent = "This position has appeared twice. A third time is a draw.";
    sub.classList.add("warn");
  } else if (game.history().length === 0) {
    sub.textContent = "Drag a piece, or tap it and then a highlighted square.";
  } else {
    sub.textContent = `Move ${Math.floor(game.history().length / 2) + 1}. You are playing ${sideName(human)}.`;
  }
}

// Scores the shown position on the panel worker, unless it is cached or already being scored.
function requestPanelAnalysis(fen: string): void {
  if (panelTimer) clearTimeout(panelTimer);
  panelTimer = setTimeout(
    () => {
      panelTimer = null;
      if (analyses.has(fen) || analysing.has(fen)) return;
      analysing.add(fen);
      void analyseForPanel(fen).then((result) => {
        analysing.delete(fen);
        analyses.set(fen, withSan(result, fen));
        if (shownPosition().fen() === fen) renderPanel(shownPosition());
      });
    },
    viewPly === null ? 0 : REVIEW_DEBOUNCE_MS,
  );
}

function renderPanel(view: Chess): void {
  const label = $("panel-label");
  const headMove = $("decision-move");
  const caption = $("decision-caption");
  const note = $("panel-note");
  const list = $("moves");
  const foot = $("panel-foot");
  const fen = view.fen();
  const mover = view.turn();
  const history = game.history({ verbose: true });
  const shownPly = viewPly ?? history.length;
  const live = viewPly === null;
  label.textContent = `${mover === human ? "Your options" : "The net's options"} · ${sideName(mover)}`;

  const clear = (message: string): void => {
    headMove.textContent = "—";
    caption.textContent = "";
    note.textContent = message;
    list.innerHTML = "";
    foot.textContent = "";
  };
  const legalCount = view.moves({ verbose: true }).length;
  if (legalCount === 0 || (live && outcome())) {
    clear("The game is over here — there are no moves to score.");
    return;
  }

  const result = analyses.get(fen);
  if (!result) {
    if (!analysing.has(fen)) requestPanelAnalysis(fen);
    headMove.textContent = "…";
    caption.textContent = `scoring ${legalCount} legal moves`;
    note.textContent = "";
    list.innerHTML = "";
    foot.textContent = "";
    return;
  }

  const played = shownPly < history.length ? history[shownPly] : null;
  const playedUci = played ? uciOf(played) : null;
  const playedIndex = playedUci ? result.moves.findIndex((m) => m.uci === playedUci) : -1;
  const whoFor = mover === human ? "for you" : "for the net";
  headMove.textContent = result.best.san ?? result.best.uci;
  caption.textContent = `its pick ${whoFor} · ${fmtPawns(result.best.score)}`;

  if (played && playedIndex >= 0) {
    const who = played.color === human ? "You played" : "The net played";
    const entry = result.moves[playedIndex];
    const rank = playedIndex === 0 ? "its top pick" : `ranked #${playedIndex + 1} of ${result.moves.length}`;
    note.textContent = `${who} ${entry.san ?? entry.uci} (${fmtPawns(entry.score)}) — ${rank}.`;
  } else if (live && history.length && history[history.length - 1].color !== human) {
    const lastMove = history[history.length - 1];
    const before = analyses.get(positionAtPly(history.length - 1).fen());
    const entry = before?.moves.find((m) => m.uci === uciOf(lastMove));
    note.textContent = entry
      ? `The net just played ${lastMove.san} (${fmtPawns(entry.score)} for it). Press ← to see everything it weighed.`
      : `The net just played ${lastMove.san}. Press ← to see everything it weighed.`;
  } else {
    note.textContent = "";
  }

  const clamp = (score: number): number => Math.max(-20, Math.min(20, score));
  const scores = result.moves.map((m) => clamp(m.score));
  const hi = Math.max(...scores);
  const lo = Math.min(...scores);
  const row = (index: number): string => {
    const m = result.moves[index];
    const width = hi === lo ? 100 : 8 + (92 * (clamp(m.score) - lo)) / (hi - lo);
    const classes = [m.uci === result.best.uci ? "chosen" : "", m.uci === playedUci ? "played" : ""].join(" ").trim();
    const score = `${m.exact ? "" : "≤ "}${fmtPawns(m.score)}`;
    return `<li class="${classes}"><span>${m.san ?? m.uci}</span><span class="bar"><span style="width:${width.toFixed(1)}%"></span></span><span class="score">${score}</span></li>`;
  };
  let rows = result.moves.slice(0, SHOWN_MOVES).map((_, i) => row(i)).join("");
  if (playedIndex >= SHOWN_MOVES) rows += `<li class="gap" aria-hidden="true">⋯</li>${row(playedIndex)}`;
  list.innerHTML = rows;

  const hiddenCount = result.moves.length - SHOWN_MOVES - (playedIndex >= SHOWN_MOVES ? 1 : 0);
  const bounded = result.moves.some((m) => !m.exact);
  foot.textContent =
    (hiddenCount > 0 ? `+ ${hiddenCount} more legal moves · ` : "") +
    (played ? "left bar marks the move played · " : "") +
    (bounded ? "≤ marks moves cut off once clearly worse · " : "") +
    `${result.nodes.toLocaleString()} positions searched · ${Math.round(result.ms)} ms`;
}

function renderSheet(): void {
  const history = game.history();
  const sheet = $("sheet");
  if (history.length === 0) {
    sheet.innerHTML = `<p class="sheet-empty">No moves yet.</p>`;
    return;
  }
  const shownPly = viewPly ?? history.length;
  const cell = (index: number): string => {
    if (index >= history.length) return "<span></span>";
    const current = index + 1 === shownPly ? " current" : "";
    return `<button type="button" class="ply${current}" data-ply="${index + 1}">${history[index]}</button>`;
  };
  let rows = "";
  for (let i = 0; i < history.length; i += 2) {
    rows += `<li><span class="num">${i / 2 + 1}.</span>${cell(i)}${cell(i + 1)}</li>`;
  }
  sheet.innerHTML = `<ol>${rows}</ol>`;
  if (viewPly === null) sheet.scrollTop = sheet.scrollHeight;
  else sheet.querySelector(".current")?.scrollIntoView({ block: "nearest" });
}

function renderNav(): void {
  const total = game.history().length;
  const shownPly = viewPly ?? total;
  $<HTMLButtonElement>("nav-first").disabled = shownPly === 0;
  $<HTMLButtonElement>("nav-prev").disabled = shownPly === 0;
  $<HTMLButtonElement>("nav-next").disabled = viewPly === null;
  $<HTMLButtonElement>("nav-last").disabled = viewPly === null;
}

function render(): void {
  const view = shownPosition();
  renderBoard(view);
  renderStatus(view);
  renderPanel(view);
  renderSheet();
  renderNav();
  $<HTMLButtonElement>("take-back").disabled = game.history().length < (human === "b" ? 2 : 1);
}

function animateMove(from: string, to: string): void {
  if (reduceMotion || viewPly !== null) return;
  const origin = buttonAt.get(from);
  const destination = buttonAt.get(to);
  const piece = destination?.querySelector<HTMLElement>(".piece");
  if (!origin || !destination || !piece) return;
  const a = origin.getBoundingClientRect();
  const b = destination.getBoundingClientRect();
  piece.classList.add("moving");
  const animation = piece.animate(
    [{ transform: `translate(${a.left - b.left}px, ${a.top - b.top}px)` }, { transform: "translate(0, 0)" }],
    { duration: ANIM_MS, easing: "cubic-bezier(0.2, 0.7, 0.3, 1)" },
  );
  animation.finished.then(() => piece.classList.remove("moving")).catch(() => undefined);
}

// ------------------------------------------------------------------- sound

function ensureAudio(): void {
  if (audio || !soundOn) return;
  const Context = window.AudioContext ?? window.webkitAudioContext;
  try {
    audio = Context ? new Context() : null;
  } catch {
    audio = null;
  }
}

function playSound(move: ChessMove): void {
  if (!soundOn || !audio) return;
  const capture = move.flags.includes("c") || move.flags.includes("e");
  const seconds = capture ? 0.1 : 0.06;
  const buffer = audio.createBuffer(1, Math.floor(audio.sampleRate * seconds), audio.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / data.length, capture ? 2 : 5);
  const source = audio.createBufferSource();
  source.buffer = buffer;
  const filter = audio.createBiquadFilter();
  filter.type = "bandpass";
  filter.frequency.value = capture ? 1500 : 950;
  filter.Q.value = 1.1;
  const gain = audio.createGain();
  gain.gain.value = capture ? 0.9 : 0.7;
  source.connect(filter).connect(gain).connect(audio.destination);
  source.start();
}

// -------------------------------------------------------------------- play

function clearSelection(): void {
  selected = null;
  targets = [];
}

function clearMarks(): void {
  marks.squares.clear();
  marks.arrows.clear();
}

function select(square: string): boolean {
  if (yourTurn()) {
    const piece = game.get(square);
    if (!piece || piece.color !== human) return false;
    selected = square;
    targets = [...new Set(game.moves({ square, verbose: true }).map((m) => m.to))];
    return true;
  }
  if (premoveTime() && !pendingPromotion) {
    const pieces = displayPieces();
    const piece = pieces.get(square);
    if (!piece || piece.color !== human) return false;
    selected = square;
    targets = premoveTargets(pieces, square);
    return true;
  }
  return false;
}

/** Plays or queues from → to; returns false if it is neither a move nor a premove. */
function attempt(from: string, to: string, animate: boolean): boolean {
  if (yourTurn()) {
    const options = game.moves({ square: from, verbose: true }).filter((m) => m.to === to);
    if (!options.length) return false;
    clearSelection();
    if (options.some((m) => m.promotion)) askPromotion(to, options);
    else playHuman(options[0], animate);
    return true;
  }
  if (premoveTime() && premoveTargets(displayPieces(), from).includes(to)) {
    premoves.push({ from, to });
    clearSelection();
    render();
    return true;
  }
  return false;
}

function playHuman(move: ChessMove, animate: boolean): void {
  const played = game.move({ from: move.from, to: move.to, promotion: move.promotion });
  if (!played) return;
  keys.push(positionKey(game));
  clearMarks();
  playSound(played);
  if (!outcome()) requestNetMove();
  render();
  if (animate) animateMove(played.from, played.to);
}

function requestNetMove(): void {
  thinking = true;
  const token = generation;
  const fen = game.fen();
  const started = performance.now();
  const cached = analyses.get(fen);
  if (!cached) analysing.add(fen);
  const search = cached ? Promise.resolve(cached) : analyseForMove(fen);
  render();
  void search.then((raw) => {
    const result = cached ?? withSan(raw, fen);
    if (!cached) {
      analysing.delete(fen);
      analyses.set(fen, result);
      if (shownPosition().fen() === fen) renderPanel(shownPosition());
    }
    const wait = Math.max(0, MIN_REPLY_MS - (performance.now() - started));
    setTimeout(() => {
      if (token === generation) applyNetMove(result, token);
    }, wait);
  });
}

function applyNetMove(result: Analysis, token: number): void {
  const best = result.best;
  const played = game.move({ from: best.from, to: best.to, promotion: best.promotion ?? undefined });
  if (!played) return;
  keys.push(positionKey(game));
  thinking = false;
  playSound(played);

  const queued = premoves.shift();
  let premove: ChessMove | null = null;
  if (queued && !outcome()) {
    const legal = game.moves({ square: queued.from, verbose: true }).filter((m) => m.to === queued.to);
    premove = legal.find((m) => m.promotion === "q") ?? legal[0] ?? null;
  }
  if (queued && !premove) premoves = [];

  if (selected && !drag) {
    const square = selected;
    clearSelection();
    select(square);
  }
  render();
  animateMove(played.from, played.to);
  if (premove) {
    const move = premove;
    setTimeout(() => {
      if (token === generation && yourTurn()) playHuman(move, true);
    }, reduceMotion ? 0 : ANIM_MS);
  }
}

function askPromotion(to: string, options: ChessMove[]): void {
  pendingPromotion = options;
  const col = flipped ? 7 - fileIndex(to) : fileIndex(to);
  const onTop = (to[1] === "8") !== flipped;
  const strip = $("promo");
  strip.style.left = `${col * 12.5}%`;
  strip.classList.toggle("from-bottom", !onTop);
  const choices: ChessPieceType[] = ["q", "n", "r", "b"];
  strip.innerHTML =
    choices
      .map((type) => `<button type="button" data-type="${type}" aria-label="Promote to ${NAMES[type]}"><span class="piece ${human}" data-type="${type}">${GLYPH[type]}${TEXT_STYLE}</span></button>`)
      .join("") + `<button type="button" class="promo-cancel" data-type="" aria-label="Cancel promotion">×</button>`;
  strip.hidden = false;
  render();
  strip.querySelector("button")?.focus();
}

function finishPromotion(type: string | null): void {
  const options = pendingPromotion;
  pendingPromotion = null;
  $("promo").hidden = true;
  const move = type && options ? options.find((m) => m.promotion === type) : undefined;
  if (move) playHuman(move, false);
  else render();
}

// ------------------------------------------------------------------ review

function goToPly(ply: number | null): void {
  const total = game.history().length;
  const next = ply === null || ply >= total ? null : Math.max(0, ply);
  if (next === viewPly) return;
  if (drag) {
    drag.el.remove();
    drag = null;
  }
  if (pendingPromotion) {
    pendingPromotion = null;
    $("promo").hidden = true;
  }
  clearSelection();
  viewPly = next;
  render();
}

function stepReview(key: string): boolean {
  const total = game.history().length;
  const current = viewPly ?? total;
  switch (key) {
    case "ArrowLeft":
      goToPly(Math.max(0, current - 1));
      return true;
    case "ArrowRight":
      goToPly(current + 1);
      return true;
    case "ArrowUp":
    case "Home":
      goToPly(0);
      return true;
    case "ArrowDown":
    case "End":
      goToPly(null);
      return true;
    default:
      return false;
  }
}

// ----------------------------------------------------------------- pointer

function squareAt(x: number, y: number): string | null {
  const rect = $("board").getBoundingClientRect();
  if (x < rect.left || y < rect.top || x >= rect.right || y >= rect.bottom) return null;
  const col = Math.floor(((x - rect.left) / rect.width) * 8);
  const row = Math.floor(((y - rect.top) / rect.height) * 8);
  return FILES[flipped ? 7 - col : col] + (flipped ? row + 1 : 8 - row);
}

function floatPiece(piece: ChessPiece): { el: HTMLDivElement; size: number } {
  const size = $("board").getBoundingClientRect().width / 8;
  const el = document.createElement("div");
  el.className = "float-piece";
  el.style.width = `${size}px`;
  el.style.height = `${size}px`;
  el.style.fontSize = `${size * 0.76}px`;
  el.innerHTML = `<span class="piece ${piece.color}" data-type="${piece.type}">${GLYPH[piece.type]}${TEXT_STYLE}</span>`;
  document.body.appendChild(el);
  return { el, size };
}

function moveFloat(d: Drag, x: number, y: number): void {
  d.el.style.transform = `translate(${x - d.size / 2}px, ${y - d.size / 2}px) scale(1.06)`;
}

function snapBack(el: HTMLDivElement, from: string): void {
  const origin = buttonAt.get(from);
  if (reduceMotion || !origin) {
    el.remove();
    render();
    return;
  }
  const rect = origin.getBoundingClientRect();
  returning = from;
  render();
  const animation = el.animate(
    [{ transform: el.style.transform }, { transform: `translate(${rect.left}px, ${rect.top}px) scale(1)` }],
    { duration: 140, easing: "ease-out", fill: "forwards" },
  );
  animation.finished
    .catch(() => undefined)
    .then(() => {
      el.remove();
      if (returning === from) {
        returning = null;
        render();
      }
    });
}

function onPointerDown(event: PointerEvent): void {
  if ((event.target as Element).closest("#promo")) return;
  ensureAudio();
  if (event.button === 2) {
    onRightDown(event);
    return;
  }
  if (event.button !== 0) return;
  event.preventDefault();
  if (viewPly !== null) {
    goToPly(null);
    return;
  }
  if (pendingPromotion) {
    finishPromotion(null);
    return;
  }
  const square = squareAt(event.clientX, event.clientY);
  if (!square) return;
  clearMarks();

  if (selected && selected !== square && targets.includes(square)) {
    if (!attempt(selected, square, true)) {
      clearSelection();
      render();
    }
    return;
  }
  const wasSelected = selected === square;
  if (select(square)) {
    const piece = displayPieces().get(square);
    if (!piece) return;
    $("board").setPointerCapture(event.pointerId);
    const { el, size } = floatPiece(piece);
    drag = { from: square, pointerId: event.pointerId, wasSelected, moved: false, startX: event.clientX, startY: event.clientY, hover: square, el, size };
    moveFloat(drag, event.clientX, event.clientY);
    render();
    return;
  }
  premoves = [];
  clearSelection();
  render();
}

function onPointerMove(event: PointerEvent): void {
  if (!drag || event.pointerId !== drag.pointerId) return;
  if (!drag.moved && Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) > 4) drag.moved = true;
  moveFloat(drag, event.clientX, event.clientY);
  const hover = squareAt(event.clientX, event.clientY);
  if (hover !== drag.hover) {
    if (drag.hover) buttonAt.get(drag.hover)?.classList.remove("hover");
    drag.hover = hover;
    if (hover) buttonAt.get(hover)?.classList.add("hover");
  }
}

function onPointerUp(event: PointerEvent): void {
  if (rightStart && event.pointerId === rightStart.pointerId) {
    onRightUp(event);
    return;
  }
  if (!drag || event.pointerId !== drag.pointerId) return;
  const { from, moved, wasSelected, el } = drag;
  const to = squareAt(event.clientX, event.clientY);
  drag = null;
  if (to && to !== from) {
    if (attempt(from, to, false)) {
      el.remove();
      return;
    }
    clearSelection();
    snapBack(el, from);
    return;
  }
  el.remove();
  if (wasSelected && !moved) clearSelection(); // a second click on the same piece deselects it
  render();
}

function onPointerCancel(event: PointerEvent): void {
  if (rightStart && event.pointerId === rightStart.pointerId) rightStart = null;
  if (!drag || event.pointerId !== drag.pointerId) return;
  const { from, el } = drag;
  drag = null;
  clearSelection();
  snapBack(el, from);
}

function onRightDown(event: PointerEvent): void {
  event.preventDefault();
  if (drag) {
    const { from, el } = drag;
    drag = null;
    snapBack(el, from);
  }
  if (premoves.length || selected) {
    premoves = [];
    clearSelection();
    render();
  }
  const square = squareAt(event.clientX, event.clientY);
  if (!square) return;
  rightStart = { square, pointerId: event.pointerId };
  $("board").setPointerCapture(event.pointerId);
}

function onRightUp(event: PointerEvent): void {
  const start = rightStart;
  rightStart = null;
  const end = squareAt(event.clientX, event.clientY);
  if (!start || !end) return;
  if (end === start.square) {
    if (marks.squares.has(end)) marks.squares.delete(end);
    else marks.squares.add(end);
  } else {
    const key = start.square + end;
    if (marks.arrows.has(key)) marks.arrows.delete(key);
    else marks.arrows.set(key, { from: start.square, to: end });
  }
  render();
}

function keyboardSquare(square: string): void {
  if (viewPly !== null) {
    goToPly(null);
    return;
  }
  if (pendingPromotion) return;
  if (selected && selected !== square && targets.includes(square)) {
    if (!attempt(selected, square, true)) clearSelection();
  } else if (selected === square || !select(square)) {
    clearSelection();
  }
  render();
}

// ---------------------------------------------------------------- controls

function resetInteraction(): void {
  generation++;
  thinking = false;
  premoves = [];
  pendingPromotion = null;
  viewPly = null;
  $("promo").hidden = true;
  clearSelection();
  clearMarks();
}

function newGame(): void {
  resetInteraction();
  human = $<HTMLInputElement>("side-black").checked ? "b" : "w";
  flipped = human === "b";
  game.reset();
  rebuildKeys();
  if (human === "b") requestNetMove();
  else render();
}

function takeBack(): void {
  if (game.history().length < (human === "b" ? 2 : 1)) return;
  resetInteraction();
  let undone = 0;
  // Undo the net's reply (if any) and your last move, stopping on your turn.
  while (game.history().length && (game.turn() !== human || undone === 0)) {
    game.undo();
    undone++;
  }
  rebuildKeys();
  if (!outcome() && game.turn() !== human) requestNetMove();
  else render();
}

function renderSpec(): void {
  const meta = engine.meta;
  const dev = meta.dev.L1 !== undefined ? `L1 ${meta.dev.L1}% · L2 ${meta.dev.L2}%` : "—";
  const rows: [string, string][] = [
    ["Net", meta.run],
    ["Parameters", meta.params.toLocaleString()],
    ["Search", "depth 1 + captures"],
    ["Dev score", dev],
  ];
  $("spec").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

function start(state: SavedState): void {
  engine = createEngine(window.NET_META, window.NET_WEIGHTS_B64);
  analyseForMove = createAnalyser();
  analyseForPanel = createAnalyser();
  game = new Chess();
  if (state.pgn) game.load_pgn(state.pgn);
  human = state.human ?? "w";
  flipped = state.flipped ?? human === "b";
  soundOn = state.soundOn !== false;
  $<HTMLInputElement>(human === "b" ? "side-black" : "side-white").checked = true;
  rebuildKeys();
  buildBoard();
  renderSpec();
  void centerPieceGlyphs();

  const board = $("board");
  board.addEventListener("pointerdown", onPointerDown);
  board.addEventListener("pointermove", onPointerMove);
  board.addEventListener("pointerup", onPointerUp);
  board.addEventListener("pointercancel", onPointerCancel);
  board.addEventListener("contextmenu", (event) => event.preventDefault());
  $("promo").addEventListener("click", (event) => {
    const button = (event.target as Element).closest<HTMLButtonElement>("button");
    if (button) finishPromotion(button.dataset.type || null);
  });
  $("sheet").addEventListener("click", (event) => {
    const button = (event.target as Element).closest<HTMLButtonElement>("button[data-ply]");
    if (button) goToPly(Number(button.dataset.ply));
  });
  $("nav-first").addEventListener("click", () => goToPly(0));
  $("nav-prev").addEventListener("click", () => stepReview("ArrowLeft"));
  $("nav-next").addEventListener("click", () => stepReview("ArrowRight"));
  $("nav-last").addEventListener("click", () => goToPly(null));
  $("new-game").addEventListener("click", newGame);
  $("take-back").addEventListener("click", takeBack);
  $("flip").addEventListener("click", () => {
    flipped = !flipped;
    if (pendingPromotion) finishPromotion(null);
    else render();
  });

  const soundButton = $("sound");
  const syncSound = (): void => {
    soundButton.textContent = soundOn ? "Sound on" : "Sound off";
    soundButton.setAttribute("aria-pressed", String(soundOn));
  };
  syncSound();
  soundButton.addEventListener("click", () => {
    soundOn = !soundOn;
    if (soundOn) ensureAudio();
    syncSound();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      if (pendingPromotion) finishPromotion(null);
      else if (premoves.length || selected || marks.squares.size || marks.arrows.size) {
        premoves = [];
        clearSelection();
        clearMarks();
        render();
      }
      return;
    }
    const target = event.target as HTMLElement | null;
    if (event.altKey || event.ctrlKey || event.metaKey || target?.tagName === "INPUT") return;
    if (stepReview(event.key)) event.preventDefault();
  });

  window.claude?.hot?.snapshot?.(() => ({ pgn: game.pgn(), human, flipped, soundOn }));

  if (!outcome() && game.turn() !== human) requestNetMove();
  else render();
}

const hot = window.claude?.hot;
if (hot?.ready) hot.ready(start);
else start(hot?.data ?? {});
