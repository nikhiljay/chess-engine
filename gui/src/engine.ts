// The eval net and the harness's move search.
//
// Mirrors engine/src/nn.rs: the same 1540-input dual-perspective encoding, the
// same network (shared feature transformer → clamp → head), and the same
// depth-1 search with quiescence — captures and promotions in MVV-LVA order,
// every move when in check, en passant not treated as a capture, and castling
// treated as a rook capture (the harness's board library encodes castling as
// the king taking its own rook). A small 0x88 move generator keeps the search
// fast enough to play in a browser.

export const MATE = 100000;

export interface TensorInfo {
  name: string;
  shape: number[];
  offset: number;
  length: number;
}

export interface NetMeta {
  run: string;
  params: number;
  hidden: number;
  head: number[];
  dev: Record<string, number>;
  tensors: TensorInfo[];
}

export interface ScoredMove {
  from: string;
  to: string;
  promotion: string | null;
  uci: string;
  /** Pawns from the mover's perspective; ±MATE for a forced mate. */
  score: number;
  /** False when the move was only proven worse than the best (score is then an upper bound). */
  exact: boolean;
  san?: string;
}

export interface Analysis {
  best: ScoredMove;
  moves: ScoredMove[];
  nodes: number;
  evaluations: number;
  ms: number;
}

export interface Engine {
  meta: NetMeta;
  /** Eval in pawns from the side to move's perspective. */
  evaluate(fen: string): number;
  analyse(fen: string, nodeBudget?: number): Analysis;
  perft(fen: string, depth: number): number;
}

const WHITE = 0;
const BLACK = 1;
const PAWN = 1;
const KNIGHT = 2;
const BISHOP = 3;
const ROOK = 4;
const QUEEN = 5;
const KING = 6;
const VALUE = [0, 1, 3, 3, 5, 9, 100];
const PROMO_CHAR: Record<number, string> = { [QUEEN]: "q", [ROOK]: "r", [BISHOP]: "b", [KNIGHT]: "n" };
const FEN_PIECE: Record<string, number> = { p: PAWN, n: KNIGHT, b: BISHOP, r: ROOK, q: QUEEN, k: KING };
const KNIGHT_STEPS = [33, 31, 18, 14, -33, -31, -18, -14];
const KING_STEPS = [1, 15, 16, 17, -1, -15, -16, -17];
const BISHOP_STEPS = [15, 17, -15, -17];
const ROOK_STEPS = [1, 16, -1, -16];
const PROMOTIONS = [QUEEN, ROOK, BISHOP, KNIGHT];
const CASTLE_WK = 1;
const CASTLE_WQ = 2;
const CASTLE_BK = 4;
const CASTLE_BQ = 8;
const F_CAPTURE = 1;
const F_EP = 2;
const F_CASTLE = 4;
const F_DOUBLE = 8;

// Castling rights kept when a move touches a square (rook home squares).
const CASTLE_KEEP: number[] = new Array(128).fill(15);
CASTLE_KEEP[0] = 15 & ~CASTLE_WQ;
CASTLE_KEEP[7] = 15 & ~CASTLE_WK;
CASTLE_KEEP[112] = 15 & ~CASTLE_BQ;
CASTLE_KEEP[119] = 15 & ~CASTLE_BK;

interface Move {
  from: number;
  to: number;
  piece: number;
  captured: number;
  flags: number;
  promo: number;
}

interface Undo {
  captured: number;
  castling: number;
  ep: number;
  halfmove: number;
}

const squareName = (sq: number): string => "abcdefgh"[sq & 7] + ((sq >> 4) + 1);
const moveUci = (m: Move): string => squareName(m.from) + squareName(m.to) + (m.promo ? PROMO_CHAR[m.promo] : "");

class Position {
  board = new Int8Array(128);
  kings: [number, number] = [0, 0];
  turn = WHITE;
  castling = 0;
  ep = -1;
  halfmove = 0;
  private undoStack: Undo[] = [];

  constructor(fen: string) {
    const [placement, turn, castling = "-", ep = "-", halfmove = "0"] = fen.trim().split(/\s+/);
    let rank = 7;
    let file = 0;
    for (const ch of placement) {
      if (ch === "/") {
        rank--;
        file = 0;
      } else if (ch >= "1" && ch <= "8") {
        file += Number(ch);
      } else {
        const type = FEN_PIECE[ch.toLowerCase()];
        const color = ch === ch.toLowerCase() ? BLACK : WHITE;
        const sq = rank * 16 + file;
        this.board[sq] = type | (color << 3);
        if (type === KING) this.kings[color] = sq;
        file++;
      }
    }
    this.turn = turn === "b" ? BLACK : WHITE;
    this.castling =
      (castling.includes("K") ? CASTLE_WK : 0) |
      (castling.includes("Q") ? CASTLE_WQ : 0) |
      (castling.includes("k") ? CASTLE_BK : 0) |
      (castling.includes("q") ? CASTLE_BQ : 0);
    this.ep = ep !== "-" ? (ep.charCodeAt(1) - 49) * 16 + (ep.charCodeAt(0) - 97) : -1;
    this.halfmove = parseInt(halfmove, 10) || 0;
  }

  attacked(sq: number, by: number): boolean {
    const b = this.board;
    const pawn = PAWN | (by << 3);
    const behind = by === WHITE ? -1 : 1;
    for (const d of [15 * behind, 17 * behind]) {
      const s = sq + d;
      if (!(s & 0x88) && b[s] === pawn) return true;
    }
    const knight = KNIGHT | (by << 3);
    for (const d of KNIGHT_STEPS) {
      const s = sq + d;
      if (!(s & 0x88) && b[s] === knight) return true;
    }
    const king = KING | (by << 3);
    for (const d of KING_STEPS) {
      const s = sq + d;
      if (!(s & 0x88) && b[s] === king) return true;
    }
    const queen = QUEEN | (by << 3);
    const bishop = BISHOP | (by << 3);
    const rook = ROOK | (by << 3);
    for (const d of BISHOP_STEPS) {
      for (let s = sq + d; !(s & 0x88); s += d) {
        const p = b[s];
        if (p) {
          if (p === bishop || p === queen) return true;
          break;
        }
      }
    }
    for (const d of ROOK_STEPS) {
      for (let s = sq + d; !(s & 0x88); s += d) {
        const p = b[s];
        if (p) {
          if (p === rook || p === queen) return true;
          break;
        }
      }
    }
    return false;
  }

  inCheck(): boolean {
    return this.attacked(this.kings[this.turn], 1 - this.turn);
  }

  pseudoMoves(): Move[] {
    const moves: Move[] = [];
    const b = this.board;
    const us = this.turn;
    const them = 1 - us;
    const push = (from: number, to: number, piece: number, captured: number, flags: number, promo: number): void => {
      moves.push({ from, to, piece, captured, flags, promo });
    };
    for (let from = 0; from < 120; from++) {
      if (from & 0x88) {
        from += 7;
        continue;
      }
      const p = b[from];
      if (!p || p >> 3 !== us) continue;
      const type = p & 7;
      if (type === PAWN) {
        const dir = us === WHITE ? 16 : -16;
        const promoRank = us === WHITE ? 7 : 0;
        const one = from + dir;
        if (!(one & 0x88) && !b[one]) {
          if (one >> 4 === promoRank) {
            for (const promo of PROMOTIONS) push(from, one, PAWN, 0, 0, promo);
          } else {
            push(from, one, PAWN, 0, 0, 0);
            if (from >> 4 === (us === WHITE ? 1 : 6) && !b[one + dir]) push(from, one + dir, PAWN, 0, F_DOUBLE, 0);
          }
        }
        for (const to of [one - 1, one + 1]) {
          if (to & 0x88) continue;
          const target = b[to];
          if (target && target >> 3 === them) {
            if (to >> 4 === promoRank) {
              for (const promo of PROMOTIONS) push(from, to, PAWN, target & 7, F_CAPTURE, promo);
            } else {
              push(from, to, PAWN, target & 7, F_CAPTURE, 0);
            }
          } else if (to === this.ep) {
            push(from, to, PAWN, PAWN, F_EP, 0);
          }
        }
        continue;
      }
      const slides = type === BISHOP || type === ROOK || type === QUEEN;
      const steps = type === KNIGHT ? KNIGHT_STEPS : type === BISHOP ? BISHOP_STEPS : type === ROOK ? ROOK_STEPS : KING_STEPS;
      for (const d of steps) {
        for (let to = from + d; !(to & 0x88); to += d) {
          const target = b[to];
          if (target) {
            if (target >> 3 === them) push(from, to, type, target & 7, F_CAPTURE, 0);
            break;
          }
          push(from, to, type, 0, 0, 0);
          if (!slides) break;
        }
      }
      if (type === KING) this.castlingMoves(from, push);
    }
    return moves;
  }

  private castlingMoves(from: number, push: (from: number, to: number, piece: number, captured: number, flags: number, promo: number) => void): void {
    const b = this.board;
    const us = this.turn;
    const them = 1 - us;
    const base = us === WHITE ? 0 : 112;
    const kingside = us === WHITE ? CASTLE_WK : CASTLE_BK;
    const queenside = us === WHITE ? CASTLE_WQ : CASTLE_BQ;
    if (from !== base + 4 || !(this.castling & (kingside | queenside)) || this.attacked(from, them)) return;
    const rook = ROOK | (us << 3);
    if (
      this.castling & kingside && !b[base + 5] && !b[base + 6] && b[base + 7] === rook &&
      !this.attacked(base + 5, them) && !this.attacked(base + 6, them)
    ) {
      push(from, base + 6, KING, 0, F_CASTLE, 0);
    }
    if (
      this.castling & queenside && !b[base + 3] && !b[base + 2] && !b[base + 1] && b[base] === rook &&
      !this.attacked(base + 3, them) && !this.attacked(base + 2, them)
    ) {
      push(from, base + 2, KING, 0, F_CASTLE, 0);
    }
  }

  make(m: Move): void {
    const b = this.board;
    const us = this.turn;
    const piece = b[m.from];
    this.undoStack.push({ captured: b[m.to], castling: this.castling, ep: this.ep, halfmove: this.halfmove });
    b[m.to] = m.promo ? m.promo | (us << 3) : piece;
    b[m.from] = 0;
    if (m.flags & F_EP) b[m.to + (us === WHITE ? -16 : 16)] = 0;
    if (m.flags & F_CASTLE) {
      const base = us === WHITE ? 0 : 112;
      if (m.to === base + 6) {
        b[base + 5] = b[base + 7];
        b[base + 7] = 0;
      } else {
        b[base + 3] = b[base];
        b[base] = 0;
      }
    }
    if ((piece & 7) === KING) {
      this.kings[us] = m.to;
      this.castling &= us === WHITE ? ~(CASTLE_WK | CASTLE_WQ) : ~(CASTLE_BK | CASTLE_BQ);
    }
    this.castling &= CASTLE_KEEP[m.from] & CASTLE_KEEP[m.to];
    this.ep = m.flags & F_DOUBLE ? m.from + (us === WHITE ? 16 : -16) : -1;
    this.halfmove = (piece & 7) === PAWN || m.flags & (F_CAPTURE | F_EP) ? 0 : this.halfmove + 1;
    this.turn = 1 - us;
  }

  unmake(m: Move): void {
    const b = this.board;
    const saved = this.undoStack.pop();
    if (!saved) throw new Error("unmake without make");
    this.turn = 1 - this.turn;
    const us = this.turn;
    const piece = m.promo ? PAWN | (us << 3) : b[m.to];
    b[m.from] = piece;
    b[m.to] = saved.captured;
    if (m.flags & F_EP) b[m.to + (us === WHITE ? -16 : 16)] = PAWN | ((1 - us) << 3);
    if (m.flags & F_CASTLE) {
      const base = us === WHITE ? 0 : 112;
      if (m.to === base + 6) {
        b[base + 7] = b[base + 5];
        b[base + 5] = 0;
      } else {
        b[base] = b[base + 3];
        b[base + 3] = 0;
      }
    }
    if ((piece & 7) === KING) this.kings[us] = m.from;
    this.castling = saved.castling;
    this.ep = saved.ep;
    this.halfmove = saved.halfmove;
  }

  /** Makes m; if it leaves the mover's king attacked, takes it back and returns false. */
  tryMake(m: Move): boolean {
    const us = this.turn;
    this.make(m);
    if (this.attacked(this.kings[us], 1 - us)) {
      this.unmake(m);
      return false;
    }
    return true;
  }

  legalMoves(): Move[] {
    const legal: Move[] = [];
    for (const m of this.pseudoMoves()) {
      if (this.tryMake(m)) {
        this.unmake(m);
        legal.push(m);
      }
    }
    return legal;
  }

  hasLegalMove(pseudo: Move[]): boolean {
    for (const m of pseudo) {
      if (this.tryMake(m)) {
        this.unmake(m);
        return true;
      }
    }
    return false;
  }

  /** Active rows of the shared feature transformer for each perspective half. */
  features(): [number[], number[]] {
    const stm: number[] = [];
    const nstm: number[] = [];
    const b = this.board;
    const turn = this.turn;
    for (let sq = 0; sq < 120; sq++) {
      if (sq & 0x88) {
        sq += 7;
        continue;
      }
      const p = b[sq];
      if (!p) continue;
      const ch = (p & 7) - 1;
      const ours = p >> 3 === turn;
      const whiteSq = (sq & 7) + (sq >> 4) * 8;
      const blackSq = (sq & 7) + (7 - (sq >> 4)) * 8;
      stm.push((ours ? ch : ch + 6) * 64 + (turn === WHITE ? whiteSq : blackSq));
      nstm.push((ours ? ch + 6 : ch) * 64 + (turn === WHITE ? blackSq : whiteSq));
    }
    const [ownK, ownQ, oppK, oppQ] =
      turn === WHITE ? [CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ] : [CASTLE_BK, CASTLE_BQ, CASTLE_WK, CASTLE_WQ];
    if (this.castling & ownK) stm.push(768);
    if (this.castling & ownQ) stm.push(769);
    if (this.castling & oppK) nstm.push(768);
    if (this.castling & oppQ) nstm.push(769);
    return [stm, nstm];
  }
}

function decodeFloat32(b64: string): Float32Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer, 0, bytes.length / 4);
}

const now = (): number => performance.now();

interface Tensor {
  shape: number[];
  data: Float32Array;
}

export function createEngine(meta: NetMeta, weightsB64: string): Engine {
  const data = decodeFloat32(weightsB64);
  const tensors = new Map<string, Tensor>();
  for (const t of meta.tensors) tensors.set(t.name, { shape: t.shape, data: data.subarray(t.offset, t.offset + t.length) });
  const tensor = (name: string): Tensor => {
    const t = tensors.get(name);
    if (!t) throw new Error(`weights are missing ${name}`);
    return t;
  };

  const H = meta.hidden;
  const ft = tensor("ft_by_feature").data;
  const ftBias = tensor("ft_bias").data;
  const layers = meta.head.map((_, i) => ({ w: tensor(`head${i}_w`), b: tensor(`head${i}_b`), relu: true }));
  layers.push({ w: tensor("out_w"), b: tensor("out_b"), relu: false });

  const accStm = new Float32Array(H);
  const accNstm = new Float32Array(H);
  const width = Math.max(2 * H, ...layers.map((l) => l.w.shape[0]));
  const bufA = new Float32Array(width);
  const bufB = new Float32Array(width);
  let nodes = 0;
  let evaluations = 0;

  function accumulate(acc: Float32Array, rows: number[]): void {
    acc.set(ftBias);
    for (const row of rows) {
      const off = row * H;
      for (let j = 0; j < H; j++) acc[j] += ft[off + j];
    }
    for (let j = 0; j < H; j++) {
      const v = acc[j];
      acc[j] = v < 0 ? 0 : v > 1 ? 1 : v;
    }
  }

  function evaluatePosition(pos: Position): number {
    evaluations++;
    const [stm, nstm] = pos.features();
    accumulate(accStm, stm);
    accumulate(accNstm, nstm);
    let x = bufA;
    let y = bufB;
    x.set(accStm, 0);
    x.set(accNstm, H);
    for (const layer of layers) {
      const [outWidth, inWidth] = layer.w.shape;
      const w = layer.w.data;
      const bias = layer.b.data;
      for (let o = 0; o < outWidth; o++) {
        let sum = bias[o];
        const off = o * inWidth;
        for (let i = 0; i < inWidth; i++) sum += w[off + i] * x[i];
        y[o] = layer.relu && sum < 0 ? 0 : sum;
      }
      [x, y] = [y, x];
    }
    return x[0];
  }

  function captures(pseudo: Move[]): Move[] {
    const victim = (m: Move): number => (m.flags & F_CAPTURE ? VALUE[m.captured] : m.flags & F_CASTLE ? VALUE[ROOK] : 0);
    const key = (m: Move): number => (m.promo ? (m.promo === QUEEN ? 9 : 3) : 0) + victim(m);
    return pseudo
      .filter((m) => m.flags & (F_CAPTURE | F_CASTLE) || m.promo)
      .sort((a, b) => key(b) - key(a) || VALUE[a.piece] - VALUE[b.piece]);
  }

  function quiescence(pos: Position, alpha: number, beta: number): number {
    nodes++;
    const pseudo = pos.pseudoMoves();
    const inCheck = pos.inCheck();
    if (!pos.hasLegalMove(pseudo)) return inCheck ? -MATE : 0;
    if (pos.halfmove >= 100) return 0;
    if (!inCheck) {
      const standPat = evaluatePosition(pos);
      if (standPat >= beta) return beta;
      if (standPat > alpha) alpha = standPat;
    }
    for (const m of inCheck ? pseudo : captures(pseudo)) {
      if (!pos.tryMake(m)) continue;
      const score = -quiescence(pos, -beta, -alpha);
      pos.unmake(m);
      if (score >= beta) return beta;
      if (score > alpha) alpha = score;
    }
    return alpha;
  }

  // Scores every legal move for the side to move. Moves are searched with a
  // full window until `nodeBudget` nodes, then with the harness's window, which
  // proves a move worse than the best so far without its exact score. The
  // chosen move is the harness's choice either way.
  function analyse(fen: string, nodeBudget = 5000): Analysis {
    const started = now();
    nodes = 0;
    evaluations = 0;
    const pos = new Position(fen);
    const ordered = pos.legalMoves().map((m) => {
      pos.make(m);
      const guess = pos.inCheck() ? Infinity : -evaluatePosition(pos);
      pos.unmake(m);
      return { m, guess };
    });
    if (ordered.length === 0) throw new Error(`no legal moves in ${fen}`);
    ordered.sort((a, b) => b.guess - a.guess);

    let alpha = -Infinity;
    let best: ScoredMove | null = null;
    const scored: ScoredMove[] = [];
    for (const { m } of ordered) {
      const full = nodes < nodeBudget;
      pos.make(m);
      const score = -quiescence(pos, -Infinity, full ? Infinity : -alpha);
      pos.unmake(m);
      const entry: ScoredMove = {
        from: squareName(m.from),
        to: squareName(m.to),
        promotion: m.promo ? PROMO_CHAR[m.promo] : null,
        uci: moveUci(m),
        score,
        exact: full || score > alpha,
      };
      scored.push(entry);
      if (score > alpha) {
        alpha = score;
        best = entry;
      }
    }
    return {
      best: best ?? scored[0],
      moves: scored.sort((a, b) => b.score - a.score),
      nodes,
      evaluations,
      ms: now() - started,
    };
  }

  function perft(fen: string, depth: number): number {
    const pos = new Position(fen);
    const walk = (d: number): number => {
      if (d === 0) return 1;
      let count = 0;
      for (const m of pos.legalMoves()) {
        pos.make(m);
        count += walk(d - 1);
        pos.unmake(m);
      }
      return count;
    };
    return walk(depth);
  }

  return {
    meta,
    evaluate: (fen) => evaluatePosition(new Position(fen)),
    analyse,
    perft,
  };
}
