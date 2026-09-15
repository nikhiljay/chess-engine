//! Teacher-versus-student diagnostic: the harness's move selection with a different evaluator.
//!
//! The search is a line-for-line port of `NnEvalBot`'s reference implementation
//! (engine/src/nn.rs `try_choose_move_sequential` + `quiescence_nn_sequential`):
//! depth 1 at the root over legal moves in generation order, quiescence over
//! captures and promotions (all evasions when in check), the same alpha-beta,
//! the same mate/draw constants, the same strictly-greater tie-break, and the
//! same fallback to the first legal move. Only the evaluator changes.
//!
//! Evaluators, all returning pawns from the side to move's perspective:
//!   net    — the ONNX model itself (used to verify this port against the harness)
//!   d9     — Stockfish at --depth, mapped exactly like our training targets:
//!            clamp(score, +-2500)/100, mate -> +-25 pawns
//!   raw    — the same search, unclamped, with mate distance preserved, to
//!            separate "the target transform loses information" from the search
//!   static — Stockfish's static evaluation (`eval`), reported for the white
//!            side and negated when Black is to move
//!
//! Modes (games for verify/compare arrive on stdin in replay_check's format,
//! one per line: `<level> <nn colour: white|black> <fen>|<moves>`):
//!   verify  — replay recorded games; this port must choose the recorded move
//!             at every net turn
//!   compare — replay recorded games; report where student and teacher differ
//!   play    — play openings x both colours against a baseline level, writing
//!             compete's JSON schema
//!
//! Usage:
//!   teacher verify --model M [--threads T] < games.txt
//!   teacher compare --model M --teacher d9 [--depth 9] < games.txt > diffs.jsonl
//!   teacher play --model M --teacher d9 --openings F --level 2 --out O.json [--stride N]

use engine::bot::Bot;
use engine::game::{GameState, Outcome};
use engine::search::capture_moves;
use engine::{format_move, BaselineBot, Board, Color, Level, Move, NnEvalBot};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

const MATE_SCORE_F: f32 = 100_000.0;
const DRAW_SCORE_F: f32 = 0.0;
const MAX_PLIES: usize = 400;
/// Our training targets clamp Stockfish scores here and give mates the same value.
const SCORE_CLAMP_CP: f32 = 2500.0;

// ---------------------------------------------------------------------------
// Terminal status (Board has no public status enum through the engine crate)
// ---------------------------------------------------------------------------

enum Status {
    /// The side to move has been checkmated.
    Lost,
    Drawn,
    Ongoing,
}

fn status_of(board: &Board) -> Status {
    match GameState::from_board(board.clone()).outcome() {
        Some(Outcome::Checkmate { .. }) => Status::Lost,
        Some(Outcome::Draw) => Status::Drawn,
        None => Status::Ongoing,
    }
}

// ---------------------------------------------------------------------------
// Evaluators
// ---------------------------------------------------------------------------

/// A position evaluator in pawns, from the side to move's perspective.
trait Evaluator {
    fn eval(&self, board: &Board) -> f32;
    /// (positions asked, answers served from cache)
    fn stats(&self) -> (u64, u64) {
        (0, 0)
    }
}

struct NetEval {
    bot: NnEvalBot,
    calls: Cell<u64>,
}

impl NetEval {
    fn load(path: &PathBuf) -> NetEval {
        let bot = NnEvalBot::load(path).expect("load model");
        NetEval { bot, calls: Cell::new(0) }
    }
}

impl Evaluator for NetEval {
    fn eval(&self, board: &Board) -> f32 {
        self.calls.set(self.calls.get() + 1);
        self.bot
            .nn_eval(&GameState::from_board(board.clone()))
            .expect("nn eval")
    }

    fn stats(&self) -> (u64, u64) {
        (self.calls.get(), 0)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum SfMode {
    /// Searched score under our training transform: clamp +-2500 cp, mate -> +-25 pawns.
    Clamped,
    /// Searched score as it comes, with mate distance preserved.
    Raw,
    /// Static evaluation, no lookahead.
    Static,
}

struct Stockfish {
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
    child: Child,
}

impl Stockfish {
    fn start(path: &str, hash_mb: u32) -> Stockfish {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("start stockfish");
        let stdin = child.stdin.take().expect("stockfish stdin");
        let reader = BufReader::new(child.stdout.take().expect("stockfish stdout"));
        let mut sf = Stockfish { stdin, reader, child };
        sf.send("uci");
        sf.read_until_prefix("uciok");
        sf.send("setoption name Threads value 1");
        sf.send(&format!("setoption name Hash value {hash_mb}"));
        sf.send("isready");
        sf.read_until_prefix("readyok");
        sf
    }

    fn send(&mut self, command: &str) {
        writeln!(self.stdin, "{command}").expect("write to stockfish");
        self.stdin.flush().expect("flush stockfish");
    }

    fn read_line(&mut self) -> String {
        let mut line = String::new();
        let n = self.reader.read_line(&mut line).expect("read stockfish");
        assert!(n > 0, "stockfish closed its output");
        line
    }

    fn read_until_prefix(&mut self, prefix: &str) -> String {
        loop {
            let line = self.read_line();
            if line.starts_with(prefix) {
                return line;
            }
        }
    }

    /// (centipawns, mate distance in moves) from the side to move, at a fixed depth.
    fn search(&mut self, fen: &str, depth: u32, fresh_hash: bool) -> (i32, i32) {
        if fresh_hash {
            self.send("ucinewgame");
            self.send("isready");
            self.read_until_prefix("readyok");
        }
        self.send(&format!("position fen {fen}"));
        self.send(&format!("go depth {depth}"));
        let mut score = (0, 0);
        loop {
            let line = self.read_line();
            if line.starts_with("bestmove") {
                return score;
            }
            if let Some(parsed) = parse_info_score(&line) {
                score = parsed;
            }
        }
    }

    /// Static evaluation in pawns, as reported for the white side.
    fn static_eval(&mut self, fen: &str) -> f32 {
        self.send(&format!("position fen {fen}"));
        self.send("eval");
        let line = self.read_until_prefix("Final evaluation");
        let token = line
            .split_whitespace()
            .nth(2)
            .unwrap_or_else(|| panic!("unparsed eval line: {line}"));
        token
            .parse::<f32>()
            .unwrap_or_else(|_| panic!("unparsed eval value: {line}"))
    }
}

impl Drop for Stockfish {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.child.wait();
    }
}

/// (centipawns, mate) from a UCI info line, from the side to move's view.
fn parse_info_score(line: &str) -> Option<(i32, i32)> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.first() != Some(&"info") || tokens.contains(&"lowerbound") || tokens.contains(&"upperbound") {
        return None;
    }
    let i = tokens.iter().position(|t| *t == "score")?;
    match (tokens.get(i + 1)?, tokens.get(i + 2)?) {
        (&"cp", value) => Some((value.parse().ok()?, 0)),
        (&"mate", value) => Some((0, value.parse().ok()?)),
        _ => None,
    }
}

struct SfEval {
    engine: RefCell<Stockfish>,
    mode: SfMode,
    depth: u32,
    fresh_hash: bool,
    cache: RefCell<HashMap<String, f32>>,
    asked: Cell<u64>,
    hits: Cell<u64>,
}

impl SfEval {
    fn start(path: &str, mode: SfMode, depth: u32, fresh_hash: bool) -> SfEval {
        SfEval {
            engine: RefCell::new(Stockfish::start(path, 16)),
            mode,
            depth,
            fresh_hash,
            cache: RefCell::new(HashMap::new()),
            asked: Cell::new(0),
            hits: Cell::new(0),
        }
    }
}

impl Evaluator for SfEval {
    fn eval(&self, board: &Board) -> f32 {
        self.asked.set(self.asked.get() + 1);
        let fen = board.to_string();
        if let Some(&cached) = self.cache.borrow().get(&fen) {
            self.hits.set(self.hits.get() + 1);
            return cached;
        }
        let mut engine = self.engine.borrow_mut();
        let pawns = match self.mode {
            SfMode::Static => {
                let white_side = engine.static_eval(&fen);
                if board.side_to_move() == Color::White {
                    white_side
                } else {
                    -white_side
                }
            }
            SfMode::Clamped => {
                let (cp, mate) = engine.search(&fen, self.depth, self.fresh_hash);
                if mate != 0 {
                    mate.signum() as f32 * (SCORE_CLAMP_CP / 100.0)
                } else {
                    cp.clamp(-(SCORE_CLAMP_CP as i32), SCORE_CLAMP_CP as i32) as f32 / 100.0
                }
            }
            SfMode::Raw => {
                let (cp, mate) = engine.search(&fen, self.depth, self.fresh_hash);
                if mate != 0 {
                    // Above any centipawn score, and nearer mates score higher.
                    mate.signum() as f32 * (1000.0 - mate.abs().min(900) as f32)
                } else {
                    cp as f32 / 100.0
                }
            }
        };
        self.cache.borrow_mut().insert(fen, pawns);
        pawns
    }

    fn stats(&self) -> (u64, u64) {
        (self.asked.get(), self.hits.get())
    }
}

// ---------------------------------------------------------------------------
// The harness search, with a pluggable evaluator
// ---------------------------------------------------------------------------

fn quiescence(evaluator: &dyn Evaluator, board: &Board, mut alpha: f32, beta: f32) -> f32 {
    match status_of(board) {
        Status::Lost => return -MATE_SCORE_F,
        Status::Drawn => return DRAW_SCORE_F,
        Status::Ongoing => {}
    }

    let in_check = !board.checkers().is_empty();

    if !in_check {
        let stand_pat = evaluator.eval(board);
        if stand_pat >= beta {
            return beta;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }
    }

    let moves = if in_check {
        let mut all = Vec::new();
        board.generate_moves(|piece_moves| {
            all.extend(piece_moves);
            false
        });
        all
    } else {
        capture_moves(board)
    };

    for mv in moves {
        let mut child = board.clone();
        child.play_unchecked(mv);
        let score = -quiescence(evaluator, &child, -beta, -alpha);
        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
        }
    }

    alpha
}

/// What one move is worth to an evaluator, searched with a full window.
fn move_value(evaluator: &dyn Evaluator, game: &GameState, mv: Move) -> f32 {
    let mut child = game.board.clone();
    child.play_unchecked(mv);
    match status_of(&child) {
        Status::Lost => MATE_SCORE_F,
        Status::Drawn => DRAW_SCORE_F,
        Status::Ongoing => -quiescence(evaluator, &child, f32::NEG_INFINITY, f32::INFINITY),
    }
}

/// Depth-1 root search. Returns the chosen move and its score.
fn choose_move(evaluator: &dyn Evaluator, game: &GameState) -> Option<(Move, f32)> {
    let legal = game.legal_moves();
    if legal.is_empty() {
        return None;
    }

    let mut best: Option<Move> = None;
    let mut alpha = f32::NEG_INFINITY;

    for &mv in &legal {
        let mut child = game.board.clone();
        child.play_unchecked(mv);
        let eval = match status_of(&child) {
            Status::Lost => MATE_SCORE_F,
            Status::Drawn => DRAW_SCORE_F,
            Status::Ongoing => -quiescence(evaluator, &child, f32::NEG_INFINITY, -alpha),
        };
        if eval > alpha {
            alpha = eval;
            best = Some(mv);
        }
        if eval >= MATE_SCORE_F {
            break;
        }
    }

    match best {
        Some(mv) => Some((mv, alpha)),
        None => legal.into_iter().next().map(|mv| (mv, alpha)),
    }
}

// ---------------------------------------------------------------------------
// Game running (mirrors cli/src/compete.rs run_game)
// ---------------------------------------------------------------------------

struct MoveRecord {
    uci: String,
    fen: String,
    side: &'static str,
}

struct GameResult {
    points: f64,
    reason: &'static str,
    outcome: &'static str,
    starting_fen: String,
    moves: Vec<MoveRecord>,
}

fn run_game(
    evaluator: &dyn Evaluator,
    baseline: &BaselineBot,
    starting_fen: &str,
    net_is_white: bool,
) -> GameResult {
    baseline.reset();
    let mut game = GameState::from_fen(starting_fen).expect("valid opening FEN");
    let net_color = if net_is_white { Color::White } else { Color::Black };
    let mut moves: Vec<MoveRecord> = Vec::new();

    loop {
        if game.is_game_over() {
            let (points, outcome, reason) = match game.outcome().unwrap_or(Outcome::Draw) {
                Outcome::Checkmate { winner } if winner == net_color => (1.0, "win", "checkmate"),
                Outcome::Checkmate { .. } => (0.0, "loss", "checkmate"),
                Outcome::Draw if game.is_threefold_repetition() => (0.5, "draw", "repetition"),
                Outcome::Draw => (0.5, "draw", "draw"),
            };
            return GameResult { points, reason, outcome, starting_fen: starting_fen.to_string(), moves };
        }
        if moves.len() >= MAX_PLIES {
            return GameResult {
                points: 0.5,
                reason: "max_plies",
                outcome: "draw",
                starting_fen: starting_fen.to_string(),
                moves,
            };
        }

        let side = game.side_to_move();
        let chosen = if side == net_color {
            choose_move(evaluator, &game).map(|(mv, _)| mv)
        } else {
            baseline.choose_move(&game)
        };

        match chosen {
            Some(mv) => {
                let uci = format_move(mv);
                game.make_move(mv);
                moves.push(MoveRecord {
                    uci,
                    fen: game.board.to_string(),
                    side: if side == Color::White { "white" } else { "black" },
                });
            }
            None => {
                // compete scores a forfeit as a loss for the side that could not move.
                let points = if side == net_color { 0.0 } else { 1.0 };
                let outcome = if side == net_color { "loss" } else { "win" };
                return GameResult {
                    points,
                    reason: "forfeit",
                    outcome,
                    starting_fen: starting_fen.to_string(),
                    moves,
                };
            }
        }
    }
}

fn game_to_json(opening_index: usize, color: &str, result: &GameResult) -> String {
    let moves: Vec<String> = result
        .moves
        .iter()
        .map(|m| format!(r#"{{"uci":"{}","fen":"{}","side":"{}"}}"#, m.uci, m.fen, m.side))
        .collect();
    format!(
        r#"{{"opening_index":{},"color":"{}","outcome":"{}","reason":"{}","points":{:.1},"move_count":{},"starting_fen":"{}","moves":[{}]}}"#,
        opening_index,
        color,
        result.outcome,
        result.reason,
        result.points,
        result.moves.len(),
        result.starting_fen,
        moves.join(","),
    )
}

// ---------------------------------------------------------------------------
// Recorded games (replay_check's stdin format)
// ---------------------------------------------------------------------------

struct RecordedGame {
    level: u8,
    net_white: bool,
    fen: String,
    moves: Vec<String>,
}

fn read_recorded_games() -> Vec<RecordedGame> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("read stdin");
    input
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| {
            let (head, moves) = line.split_once('|').expect("expected '<level> <colour> <fen>|<moves>'");
            let mut parts = head.splitn(3, ' ');
            let level: u8 = parts.next().and_then(|s| s.parse().ok()).expect("level");
            let net_white = parts.next().expect("colour") == "white";
            RecordedGame {
                level,
                net_white,
                fen: parts.next().expect("fen").trim().to_string(),
                moves: moves.split_whitespace().map(str::to_string).collect(),
            }
        })
        .collect()
}

/// Walk a recorded game, calling `at_net_turn` before each move the net made.
fn replay(game_record: &RecordedGame, mut at_net_turn: impl FnMut(&GameState, &str)) {
    let mut game = GameState::from_fen(&game_record.fen).expect("valid FEN");
    let net_color = if game_record.net_white { Color::White } else { Color::Black };
    for uci in &game_record.moves {
        if game.side_to_move() == net_color {
            at_net_turn(&game, uci);
        }
        let mv = game
            .legal_moves()
            .into_iter()
            .find(|m| format_move(*m) == *uci)
            .unwrap_or_else(|| panic!("recorded move {uci} is not legal"));
        game.make_move(mv);
    }
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------

/// Run `job` over `count` items on `threads` worker threads, each with its own evaluator.
fn parallel<T: Send>(
    count: usize,
    threads: usize,
    make_evaluator: &(dyn Fn() -> Box<dyn Evaluator> + Sync),
    job: &(dyn Fn(usize, &dyn Evaluator) -> T + Sync),
) -> Vec<T> {
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<T>>> = Mutex::new((0..count).map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| {
                let evaluator = make_evaluator();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= count {
                        break;
                    }
                    let value = job(i, evaluator.as_ref());
                    results.lock().expect("results lock")[i] = Some(value);
                }
            });
        }
    });
    results.into_inner().expect("results").into_iter().map(|v| v.expect("job result")).collect()
}

fn mode_verify(cfg: &Config) {
    let games = read_recorded_games();
    let model = cfg.model.clone();
    let counts = parallel(
        games.len(),
        cfg.threads,
        &|| Box::new(NetEval::load(&model)),
        &|i, evaluator| {
            let (mut checked, mut matched, mut first_mismatch) = (0usize, 0usize, None);
            replay(&games[i], |game, recorded| {
                let chosen = choose_move(evaluator, game).map(|(mv, _)| format_move(mv));
                checked += 1;
                if chosen.as_deref() == Some(recorded) {
                    matched += 1;
                } else if first_mismatch.is_none() {
                    first_mismatch = Some(format!(
                        "game {i} ply {}: recorded {recorded}, port {chosen:?} ({})",
                        game.history.len(),
                        game.board
                    ));
                }
            });
            (checked, matched, first_mismatch)
        },
    );

    let (mut checked, mut matched) = (0usize, 0usize);
    for (c, m, mismatch) in &counts {
        checked += c;
        matched += m;
        if let Some(text) = mismatch {
            println!("mismatch: {text}");
        }
    }
    println!("verify: {matched}/{checked} net moves reproduced across {} games", games.len());
    std::process::exit(if matched == checked { 0 } else { 1 });
}

fn mode_compare(cfg: &Config) {
    let games = read_recorded_games();
    let model = cfg.model.clone();
    let (sf_path, mode, depth, fresh) = (cfg.stockfish.clone(), cfg.mode, cfg.depth, cfg.fresh_hash);
    let started = Instant::now();
    let lines = parallel(
        games.len(),
        cfg.threads,
        &|| Box::new(NetEval::load(&model)),
        &|i, student| {
            let teacher = SfEval::start(&sf_path, mode, depth, fresh);
            let mut out: Vec<String> = Vec::new();
            replay(&games[i], |game, recorded| {
                let (student_mv, student_score) = choose_move(student, game).expect("student move");
                let (teacher_mv, teacher_score) = choose_move(&teacher, game).expect("teacher move");
                // What each move is worth to the teacher, so a disagreement can be priced.
                let teacher_best = move_value(&teacher, game, teacher_mv);
                let teacher_on_student = move_value(&teacher, game, student_mv);
                out.push(format!(
                    r#"{{"game":{},"ply":{},"fen":"{}","recorded":"{}","student":"{}","student_score":{:.4},"teacher":"{}","teacher_score":{:.4},"teacher_best_value":{:.4},"teacher_value_of_student_move":{:.4},"cost":{:.4},"agree":{}}}"#,
                    i,
                    game.history.len(),
                    game.board,
                    recorded,
                    format_move(student_mv),
                    student_score,
                    format_move(teacher_mv),
                    teacher_score,
                    teacher_best,
                    teacher_on_student,
                    teacher_best - teacher_on_student,
                    teacher_mv == student_mv,
                ));
            });
            let (asked, hits) = teacher.stats();
            eprintln!("game {i}: {} net turns, {asked} teacher positions ({hits} cached)", out.len());
            out
        },
    );
    let mut stdout = std::io::stdout().lock();
    let (mut turns, mut agree) = (0usize, 0usize);
    for line in lines.iter().flatten() {
        turns += 1;
        agree += usize::from(line.ends_with("\"agree\":true}"));
        writeln!(stdout, "{line}").expect("write");
    }
    eprintln!(
        "compare: {agree}/{turns} decisions agree across {} games in {:.1}s",
        games.len(),
        started.elapsed().as_secs_f64()
    );
}

fn mode_play(cfg: &Config) {
    let text = std::fs::read_to_string(&cfg.openings).expect("read openings");
    let all: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect();
    let openings: Vec<String> = all.iter().step_by(cfg.stride.max(1)).cloned().collect();
    let jobs: Vec<(usize, bool)> = (0..openings.len()).flat_map(|i| [(i, true), (i, false)]).collect();

    let level = Level::new(cfg.level).expect("level 1-4");
    let model = cfg.model.clone();
    let (sf_path, mode, depth, fresh, teacher) = (cfg.stockfish.clone(), cfg.mode, cfg.depth, cfg.fresh_hash, cfg.teacher.clone());
    let started = Instant::now();
    eprintln!(
        "teacher={teacher} level={} openings={} (stride {}) games={}",
        cfg.level,
        openings.len(),
        cfg.stride,
        jobs.len()
    );

    let results = parallel(
        jobs.len(),
        cfg.threads,
        &|| -> Box<dyn Evaluator> {
            if teacher == "net" {
                Box::new(NetEval::load(&model))
            } else {
                Box::new(SfEval::start(&sf_path, mode, depth, fresh))
            }
        },
        &|i, evaluator| {
            let (opening_index, net_white) = jobs[i];
            let baseline = BaselineBot::from_level(level);
            let before = evaluator.stats();
            let game_started = Instant::now();
            let result = run_game(evaluator, &baseline, &openings[opening_index], net_white);
            let after = evaluator.stats();
            eprintln!(
                "opening {opening_index} {} {} in {} plies, {:.1}s, {} evals",
                if net_white { "white" } else { "black" },
                result.outcome,
                result.moves.len(),
                game_started.elapsed().as_secs_f64(),
                after.0 - before.0,
            );
            let json = game_to_json(opening_index, if net_white { "white" } else { "black" }, &result);
            (result.points, json)
        },
    );

    let score: f64 = results.iter().map(|(points, _)| points).sum();
    let wins = results.iter().filter(|(p, _)| *p == 1.0).count();
    let draws = results.iter().filter(|(p, _)| *p == 0.5).count();
    let losses = results.iter().filter(|(p, _)| *p == 0.0).count();
    let games: Vec<String> = results.into_iter().map(|(_, json)| json).collect();
    let score_pct = 100.0 * score / games.len() as f64;
    let json = format!(
        r#"{{"param_count":0,"best_level":0,"levels":[{{"level":{},"name":"{}","score":{},"score_pct":{:.1},"wins":{},"draws":{},"losses":{},"passed":false,"games":[{}]}}]}}"#,
        cfg.level,
        level.name(),
        score,
        score_pct,
        wins,
        draws,
        losses,
        games.join(","),
    );
    std::fs::write(&cfg.out, json).expect("write results");
    eprintln!(
        "{teacher}: {score}/{} ({score_pct:.1}%) W{wins} D{draws} L{losses} in {:.1}s -> {}",
        games.len(),
        started.elapsed().as_secs_f64(),
        cfg.out.display(),
    );
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Config {
    mode_name: String,
    model: PathBuf,
    teacher: String,
    mode: SfMode,
    depth: u32,
    fresh_hash: bool,
    stockfish: String,
    openings: PathBuf,
    out: PathBuf,
    level: u8,
    stride: usize,
    threads: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: teacher <verify|compare|play> --model M [options]");
        std::process::exit(2);
    }
    let mut cfg = Config {
        mode_name: args[1].clone(),
        model: PathBuf::new(),
        teacher: "d9".to_string(),
        mode: SfMode::Clamped,
        depth: 9,
        fresh_hash: true,
        stockfish: "stockfish".to_string(),
        openings: PathBuf::new(),
        out: PathBuf::from("teacher_games.json"),
        level: 2,
        stride: 1,
        threads: 8,
    };
    let mut i = 2;
    while i < args.len() {
        let value = || args.get(i + 1).unwrap_or_else(|| panic!("{} needs a value", args[i])).clone();
        match args[i].as_str() {
            "--model" => cfg.model = PathBuf::from(value()),
            "--teacher" => cfg.teacher = value(),
            "--depth" => cfg.depth = value().parse().expect("depth"),
            "--stockfish" => cfg.stockfish = value(),
            "--openings" => cfg.openings = PathBuf::from(value()),
            "--out" => cfg.out = PathBuf::from(value()),
            "--level" => cfg.level = value().parse().expect("level"),
            "--stride" => cfg.stride = value().parse().expect("stride"),
            "--threads" => cfg.threads = value().parse().expect("threads"),
            "--keep-hash" => {
                cfg.fresh_hash = false;
                i += 1;
                continue;
            }
            other => panic!("unknown argument {other}"),
        }
        i += 2;
    }
    cfg.mode = match cfg.teacher.as_str() {
        "net" | "d9" => SfMode::Clamped,
        "raw" => SfMode::Raw,
        "static" => SfMode::Static,
        other => panic!("unknown teacher {other} (net, d9, raw, static)"),
    };

    match cfg.mode_name.as_str() {
        "verify" => mode_verify(&cfg),
        "compare" => mode_compare(&cfg),
        "play" => mode_play(&cfg),
        other => panic!("unknown mode {other} (verify, compare, play)"),
    }
}
