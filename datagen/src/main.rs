//! Generate training positions from games between the NN and a baseline bot.
//!
//! Games start from a few random legal plies, and each move is replaced by a
//! random legal move with some probability, so positions include the lopsided
//! and unusual situations the net reaches in real games. Output is one FEN per
//! line, deduplicated, ready for `chess_nn.label`.
//!
//! With `--loop-plies N`, games that end in threefold repetition while the net
//! thinks it is at least `--min-eval` pawns ahead also emit positions from the
//! last N plies: by default (`--emit children`) every quiet child of each
//! position the net moved from, so labels tell the net which moves make
//! progress instead of shuffling; with `--emit parents`, those positions the
//! net moved from, so their children can be grouped per parent downstream.
//!
//! With `--record FILE`, every game is also written as one JSON line — start
//! position (after the random opening plies), moves in UCI, which side the net
//! played, result and how it ended — for `chess_nn.diagnose`.
//!
//! Usage:
//!   datagen <model.onnx> --out FILE [--games N] [--threads T] [--level L]
//!           [--random-move-prob P] [--sample-prob Q] [--min-ply K] [--seed S]
//!           [--loop-plies N] [--min-eval PAWNS] [--emit children|parents]
//!           [--record FILE]

use engine::bot::Bot;
use engine::game::{GameState, Outcome};
use engine::{BaselineBot, Board, Color, Level, NnEvalBot};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Instant;

const MAX_PLIES: usize = 400;
const OPENING_PLIES: (usize, usize) = (2, 8);

struct Config {
    model: PathBuf,
    out: PathBuf,
    games: usize,
    threads: usize,
    level: u8,
    random_move_prob: f64,
    sample_prob: f64,
    min_ply: usize,
    seed: u64,
    loop_plies: usize,
    min_eval: f32,
    emit_parents: bool,
    record: Option<PathBuf>,
}

enum Output {
    Position(u64, String),
    Game(String),
}

/// xorshift64* — dependency-free and good enough for move sampling.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// How a finished (or abandoned) game ended, and its result string.
fn ending(game: &GameState) -> (&'static str, &'static str) {
    if game.is_threefold_repetition() {
        return ("repetition", "1/2-1/2");
    }
    match game.outcome() {
        Some(Outcome::Checkmate { winner: Color::White }) => ("checkmate", "1-0"),
        Some(Outcome::Checkmate { winner: Color::Black }) => ("checkmate", "0-1"),
        Some(Outcome::Draw) if game.board.halfmove_clock() >= 100 => ("fifty_move", "1/2-1/2"),
        Some(Outcome::Draw) => ("stalemate", "1/2-1/2"),
        None => ("max_plies", "*"),
    }
}

/// Plays one game; returns true if it emitted loop positions.
fn play_game(
    nn: &NnEvalBot,
    baseline: &BaselineBot,
    nn_white: bool,
    cfg: &Config,
    rng: &mut Rng,
    tx: &mpsc::Sender<Output>,
) -> bool {
    let mut game = GameState::new();
    let opening = OPENING_PLIES.0 + rng.below(OPENING_PLIES.1 - OPENING_PLIES.0 + 1);
    for _ in 0..opening {
        let moves = game.legal_moves();
        if moves.is_empty() {
            return false;
        }
        game.make_move(moves[rng.below(moves.len())]);
    }
    baseline.reset();
    let start_fen = game.board.to_string();
    let mut moves_uci: Vec<String> = Vec::new();
    // (ply, position) for each position the net moved from; loop mode only.
    let mut nn_positions: Vec<(usize, Board)> = Vec::new();

    for ply in opening..MAX_PLIES {
        if game.is_game_over() {
            break;
        }
        let moves = game.legal_moves();
        let nn_to_move = (game.side_to_move() == Color::White) == nn_white;
        if nn_to_move && cfg.loop_plies > 0 {
            nn_positions.push((ply, game.board.clone()));
        }
        let mv = if rng.unit() < cfg.random_move_prob {
            Some(moves[rng.below(moves.len())])
        } else if nn_to_move {
            nn.choose_move(&game)
        } else {
            baseline.choose_move(&game)
        };
        let Some(mv) = mv else { break };
        moves_uci.push(mv.to_string());
        game.make_move(mv);

        let sample = ply + 1 >= cfg.min_ply
            && game.board.checkers().is_empty()
            && !game.is_game_over()
            && rng.unit() < cfg.sample_prob;
        if sample && tx.send(Output::Position(game.board.hash(), game.board.to_string())).is_err() {
            return false;
        }
    }

    if cfg.record.is_some() {
        let (end, result) = ending(&game);
        let moves_json = moves_uci.iter().map(|m| format!("\"{m}\"")).collect::<Vec<_>>().join(",");
        let line = format!(
            "{{\"start_fen\":\"{start_fen}\",\"moves\":[{moves_json}],\"net_white\":{nn_white},\"level\":{},\"result\":\"{result}\",\"end\":\"{end}\"}}",
            cfg.level
        );
        if tx.send(Output::Game(line)).is_err() {
            return false;
        }
    }

    cfg.loop_plies > 0
        && game.is_threefold_repetition()
        && emit_loop_positions(nn, game.history.len(), &nn_positions, cfg, tx)
}

/// If the net's eval of the last position it moved from is at least
/// `cfg.min_eval` pawns, sends the positions it moved from in the last
/// `cfg.loop_plies` plies (`--emit parents`) or their quiet, ongoing
/// children (`--emit children`).
fn emit_loop_positions(
    nn: &NnEvalBot,
    end_ply: usize,
    nn_positions: &[(usize, Board)],
    cfg: &Config,
    tx: &mpsc::Sender<Output>,
) -> bool {
    let Some((_, last)) = nn_positions.last() else {
        return false;
    };
    let eval = nn.nn_eval(&GameState::from_board(last.clone())).unwrap_or(0.0);
    if eval < cfg.min_eval {
        return false;
    }
    let mut seen = HashSet::new();
    for (ply, board) in nn_positions.iter().rev() {
        if ply + cfg.loop_plies < end_ply {
            break;
        }
        if !seen.insert(board.hash()) {
            continue;
        }
        if cfg.emit_parents {
            if tx.send(Output::Position(board.hash(), board.to_string())).is_err() {
                return false;
            }
            continue;
        }
        let mut children = Vec::new();
        board.generate_moves(|moves| {
            for mv in moves {
                let mut child = board.clone();
                child.play_unchecked(mv);
                children.push(child);
            }
            false
        });
        for child in children {
            if !child.checkers().is_empty() || GameState::from_board(child.clone()).is_game_over() {
                continue;
            }
            if tx.send(Output::Position(child.hash(), child.to_string())).is_err() {
                return false;
            }
        }
    }
    true
}

fn parse<T: std::str::FromStr>(flag: &str, value: &str) -> T {
    value.parse().unwrap_or_else(|_| {
        eprintln!("invalid value for {flag}: {value}");
        std::process::exit(1)
    })
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        eprintln!("Usage: datagen <model.onnx> --out FILE [--games N] [--threads T] [--level L]");
        eprintln!("               [--random-move-prob P] [--sample-prob Q] [--min-ply K] [--seed S]");
        eprintln!("               [--loop-plies N] [--min-eval PAWNS] [--emit children|parents] [--record FILE]");
        std::process::exit(1);
    }
    let mut cfg = Config {
        model: PathBuf::from(&args[1]),
        out: PathBuf::from("positions.txt"),
        games: 1000,
        threads: 8,
        level: 1,
        random_move_prob: 0.1,
        sample_prob: 0.25,
        min_ply: 16,
        seed: 1,
        loop_plies: 0,
        min_eval: 3.0,
        emit_parents: false,
        record: None,
    };
    let mut i = 2;
    while i < args.len() {
        let flag = args[i].as_str();
        let Some(value) = args.get(i + 1) else {
            eprintln!("missing value for {flag}");
            std::process::exit(1);
        };
        match flag {
            "--out" => cfg.out = PathBuf::from(value),
            "--games" => cfg.games = parse(flag, value),
            "--threads" => cfg.threads = parse(flag, value),
            "--level" => cfg.level = parse(flag, value),
            "--random-move-prob" => cfg.random_move_prob = parse(flag, value),
            "--sample-prob" => cfg.sample_prob = parse(flag, value),
            "--min-ply" => cfg.min_ply = parse(flag, value),
            "--seed" => cfg.seed = parse(flag, value),
            "--loop-plies" => cfg.loop_plies = parse(flag, value),
            "--min-eval" => cfg.min_eval = parse(flag, value),
            "--record" => cfg.record = Some(PathBuf::from(value)),
            "--emit" => {
                cfg.emit_parents = match value.as_str() {
                    "children" => false,
                    "parents" => true,
                    _ => {
                        eprintln!("--emit must be children or parents");
                        std::process::exit(1);
                    }
                }
            }
            _ => {
                eprintln!("unknown flag {flag}");
                std::process::exit(1);
            }
        }
        i += 2;
    }
    if Level::new(cfg.level).is_none() {
        eprintln!("--level must be 1-4");
        std::process::exit(1);
    }
    cfg
}

fn main() {
    let cfg = Arc::new(parse_args());
    let games_done = Arc::new(AtomicUsize::new(0));
    let loop_games = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel::<Output>();
    let per_thread = cfg.games.div_ceil(cfg.threads);

    let mut workers = Vec::new();
    for t in 0..cfg.threads {
        let (tx, cfg, games_done, loop_games) = (tx.clone(), cfg.clone(), games_done.clone(), loop_games.clone());
        workers.push(thread::spawn(move || {
            let nn = NnEvalBot::load(&cfg.model).unwrap_or_else(|e| panic!("failed to load model: {e}"));
            let baseline = BaselineBot::from_level(Level::new(cfg.level).unwrap());
            let mut rng = Rng::new(cfg.seed ^ (t as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
            for g in 0..per_thread {
                if play_game(&nn, &baseline, (g + t) % 2 == 0, &cfg, &mut rng, &tx) {
                    loop_games.fetch_add(1, Ordering::Relaxed);
                }
                games_done.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }
    drop(tx);

    let start = Instant::now();
    let mut out = BufWriter::new(File::create(&cfg.out).expect("cannot create output file"));
    let mut record = cfg
        .record
        .as_ref()
        .map(|path| BufWriter::new(File::create(path).expect("cannot create record file")));
    let mut seen = HashSet::new();
    let (mut written, mut duplicates, mut recorded) = (0usize, 0usize, 0usize);
    for output in rx {
        match output {
            Output::Game(line) => {
                if let Some(file) = record.as_mut() {
                    writeln!(file, "{line}").expect("write failed");
                    recorded += 1;
                }
            }
            Output::Position(hash, fen) => {
                if !seen.insert(hash) {
                    duplicates += 1;
                    continue;
                }
                writeln!(out, "{fen}").expect("write failed");
                written += 1;
                if written % 100_000 == 0 {
                    eprintln!(
                        "{written} positions, {duplicates} duplicates, {} games ({} loops), {:.0}s",
                        games_done.load(Ordering::Relaxed),
                        loop_games.load(Ordering::Relaxed),
                        start.elapsed().as_secs_f64()
                    );
                }
            }
        }
    }
    out.flush().expect("flush failed");
    if let Some(file) = record.as_mut() {
        file.flush().expect("flush failed");
    }
    for worker in workers {
        worker.join().expect("worker panicked");
    }
    eprintln!(
        "done: {written} positions, {duplicates} duplicates, {recorded} games recorded, {} games ({} loops), {:.0}s",
        games_done.load(Ordering::Relaxed),
        loop_games.load(Ordering::Relaxed),
        start.elapsed().as_secs_f64()
    );
}
