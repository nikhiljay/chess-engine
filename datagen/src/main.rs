//! Generate training positions from games between the NN and a baseline bot.
//!
//! Games start from a few random legal plies, and each move is replaced by a
//! random legal move with some probability, so positions include the lopsided
//! and unusual situations the net reaches in real games. Output is one FEN per
//! line, deduplicated, ready for `chess_nn.label`.
//!
//! Usage:
//!   datagen <model.onnx> --out FILE [--games N] [--threads T] [--level L]
//!           [--random-move-prob P] [--sample-prob Q] [--min-ply K] [--seed S]

use engine::bot::Bot;
use engine::game::GameState;
use engine::{BaselineBot, Color, Level, NnEvalBot};
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

fn play_game(
    nn: &NnEvalBot,
    baseline: &BaselineBot,
    nn_white: bool,
    cfg: &Config,
    rng: &mut Rng,
    tx: &mpsc::Sender<(u64, String)>,
) {
    let mut game = GameState::new();
    let opening = OPENING_PLIES.0 + rng.below(OPENING_PLIES.1 - OPENING_PLIES.0 + 1);
    for _ in 0..opening {
        let moves = game.legal_moves();
        if moves.is_empty() {
            return;
        }
        game.make_move(moves[rng.below(moves.len())]);
    }
    baseline.reset();

    for ply in opening..MAX_PLIES {
        if game.is_game_over() {
            return;
        }
        let moves = game.legal_moves();
        let nn_to_move = (game.side_to_move() == Color::White) == nn_white;
        let mv = if rng.unit() < cfg.random_move_prob {
            Some(moves[rng.below(moves.len())])
        } else if nn_to_move {
            nn.choose_move(&game)
        } else {
            baseline.choose_move(&game)
        };
        let Some(mv) = mv else { return };
        game.make_move(mv);

        let sample = ply + 1 >= cfg.min_ply
            && game.board.checkers().is_empty()
            && !game.is_game_over()
            && rng.unit() < cfg.sample_prob;
        if sample && tx.send((game.board.hash(), game.board.to_string())).is_err() {
            return;
        }
    }
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
    let (tx, rx) = mpsc::channel::<(u64, String)>();
    let per_thread = cfg.games.div_ceil(cfg.threads);

    let mut workers = Vec::new();
    for t in 0..cfg.threads {
        let (tx, cfg, games_done) = (tx.clone(), cfg.clone(), games_done.clone());
        workers.push(thread::spawn(move || {
            let nn = NnEvalBot::load(&cfg.model).unwrap_or_else(|e| panic!("failed to load model: {e}"));
            let baseline = BaselineBot::from_level(Level::new(cfg.level).unwrap());
            let mut rng = Rng::new(cfg.seed ^ (t as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
            for g in 0..per_thread {
                play_game(&nn, &baseline, (g + t) % 2 == 0, &cfg, &mut rng, &tx);
                games_done.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }
    drop(tx);

    let start = Instant::now();
    let mut out = BufWriter::new(File::create(&cfg.out).expect("cannot create output file"));
    let mut seen = HashSet::new();
    let (mut written, mut duplicates) = (0usize, 0usize);
    for (hash, fen) in rx {
        if !seen.insert(hash) {
            duplicates += 1;
            continue;
        }
        writeln!(out, "{fen}").expect("write failed");
        written += 1;
        if written % 100_000 == 0 {
            eprintln!(
                "{written} positions, {duplicates} duplicates, {} games, {:.0}s",
                games_done.load(Ordering::Relaxed),
                start.elapsed().as_secs_f64()
            );
        }
    }
    out.flush().expect("flush failed");
    for worker in workers {
        worker.join().expect("worker panicked");
    }
    eprintln!(
        "done: {written} positions, {duplicates} duplicates, {} games, {:.0}s",
        games_done.load(Ordering::Relaxed),
        start.elapsed().as_secs_f64()
    );
}
