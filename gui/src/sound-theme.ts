// Which lichess sound set the page ships, and which file each event plays.
// Shared by the runtime (src/sound.ts) and the build, which copies only the
// active set into dist/sounds/. See sounds/README.md for provenance and licences.

export type SoundName = "move" | "capture" | "check" | "victory" | "defeat" | "draw";

export type Theme = "standard" | "sfx";

/**
 * lichess's default set has no check sound and one shared end-of-game sound:
 * in lila, `standard/Check.mp3` is a symlink to `Silence.mp3`, and `Victory`,
 * `Defeat` and `Draw` all symlink to `GenericNotify.mp3`. Mapping them here keeps
 * that behaviour without shipping the same bytes four times.
 */
export const THEMES: Record<Theme, Record<SoundName, string>> = {
  standard: {
    move: "Move.mp3",
    capture: "Capture.mp3",
    check: "Silence.mp3",
    victory: "GenericNotify.mp3",
    defeat: "GenericNotify.mp3",
    draw: "GenericNotify.mp3",
  },
  sfx: {
    move: "Move.mp3",
    capture: "Capture.mp3",
    check: "Check.mp3",
    victory: "Victory.mp3",
    defeat: "Defeat.mp3",
    draw: "Draw.mp3",
  },
};

// "standard" is what lichess plays by default; "sfx" is the AGPL-licensed set,
// which is the one to ship if this page is ever shared.
export const THEME: Theme = "standard";

export const SOUNDS = THEMES[THEME];
