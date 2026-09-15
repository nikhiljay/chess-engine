// The subset of chess.js 0.10.3 (loaded from cdnjs as the global `Chess`) that the page uses.

type ChessColor = "w" | "b";
type ChessPieceType = "p" | "n" | "b" | "r" | "q" | "k";

interface ChessPiece {
  type: ChessPieceType;
  color: ChessColor;
}

interface ChessMove {
  color: ChessColor;
  from: string;
  to: string;
  /** n normal, b double pawn push, e en passant, c capture, p promotion, k/q castling */
  flags: string;
  piece: ChessPieceType;
  san: string;
  captured?: ChessPieceType;
  promotion?: ChessPieceType;
}

declare class Chess {
  constructor(fen?: string);
  fen(): string;
  turn(): ChessColor;
  get(square: string): ChessPiece | null;
  moves(options: { verbose: true; square?: string }): ChessMove[];
  move(move: { from: string; to: string; promotion?: string }): ChessMove | null;
  undo(): ChessMove | null;
  reset(): void;
  history(): string[];
  history(options: { verbose: true }): ChessMove[];
  in_check(): boolean;
  in_checkmate(): boolean;
  in_stalemate(): boolean;
  insufficient_material(): boolean;
  pgn(): string;
  load_pgn(pgn: string): boolean;
}
