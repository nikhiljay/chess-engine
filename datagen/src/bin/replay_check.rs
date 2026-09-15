//! Replay recorded games and check that the local baseline bot chooses every baseline move.
//!
//! Evidence that a game record (e.g. a leaderboard server replay) was played
//! against the same baseline as this harness checkout: the bot is reset at the
//! start of each game, as `compete` does, and asked for a move at every
//! baseline turn before the recorded move is played.
//!
//! Input on stdin, one game per line:
//!   <level> <nn colour: white|black> <starting FEN>|<moves in the harness's UCI, space-separated>
//!
//! Usage:
//!   datagen/target/release/replay_check < replays.txt

use engine::bot::Bot;
use engine::game::GameState;
use engine::{format_move, BaselineBot, Color, Level};
use std::io::{self, BufRead};

fn main() {
    let mut all_matched = true;
    for line in io::stdin().lock().lines() {
        let line = line.expect("read stdin");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (head, moves) = line.split_once('|').expect("expected '<level> <colour> <fen>|<moves>'");
        let mut parts = head.splitn(3, ' ');
        let level: u8 = parts.next().and_then(|s| s.parse().ok()).expect("level");
        let nn_white = parts.next().expect("colour") == "white";
        let fen = parts.next().expect("fen").trim();

        let bot = BaselineBot::from_level(Level::new(level).expect("level 1-4"));
        bot.reset();
        let mut game = GameState::from_fen(fen).expect("valid FEN");
        let (mut checked, mut matched) = (0usize, 0usize);
        let mut first_mismatch: Option<String> = None;
        for (ply, uci) in moves.split_whitespace().enumerate() {
            let baseline_turn = (game.side_to_move() == Color::White) != nn_white;
            if baseline_turn {
                let chosen = bot.choose_move(&game).map(format_move);
                checked += 1;
                if chosen.as_deref() == Some(uci) {
                    matched += 1;
                } else if first_mismatch.is_none() {
                    first_mismatch = Some(format!("ply {ply}: recorded {uci}, local {chosen:?}"));
                }
            }
            let Some(mv) = game.legal_moves().into_iter().find(|m| format_move(*m) == uci) else {
                panic!("ply {ply}: recorded move {uci} is not legal here");
            };
            game.make_move(mv);
        }
        all_matched &= matched == checked;
        println!(
            "level {level} nn {} plies {}: baseline moves matched {matched}/{checked}{}",
            if nn_white { "white" } else { "black" },
            moves.split_whitespace().count(),
            first_mismatch.map(|m| format!(" (first mismatch {m})")).unwrap_or_default(),
        );
    }
    std::process::exit(if all_matched { 0 } else { 1 });
}
