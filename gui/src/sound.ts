// Sound: lichess's samples, played through Web Audio.
//
// The files are fetched as soon as the page loads and decoded once the browser
// gives us an AudioContext, which it only does after a user gesture — so the
// first move already has its sample ready. A sound that fails to load or decode
// stays silent rather than throwing; sound is decoration and must never stall
// the board.

import { SOUNDS, type SoundName } from "./sound-theme.ts";

let ctx: AudioContext | null = null;
let master: GainNode | null = null;
let enabled = true;

declare global {
  interface Window {
    webkitAudioContext?: typeof AudioContext;
  }
}

const files = [...new Set(Object.values(SOUNDS))];
const downloads = new Map<string, Promise<ArrayBuffer>>();
const buffers = new Map<string, AudioBuffer>();

/** Fetching needs no gesture, so the bytes are on their way before the first click. */
export function preloadSounds(): void {
  for (const file of files) {
    if (downloads.has(file)) continue;
    downloads.set(
      file,
      fetch(`sounds/${file}`).then((response) => {
        if (!response.ok) throw new Error(`sounds/${file}: ${response.status}`);
        return response.arrayBuffer();
      }),
    );
  }
}

/** Creates the context on the first user gesture; browsers refuse one before that. */
export function ensureAudio(): void {
  if (!enabled) return;
  if (ctx) {
    if (ctx.state === "suspended") void ctx.resume();
    return;
  }
  const Context = window.AudioContext ?? window.webkitAudioContext;
  if (!Context) return;
  try {
    ctx = new Context();
  } catch {
    ctx = null;
    return;
  }
  master = ctx.createGain();
  master.gain.value = 0.7; // lichess's own default volume
  master.connect(ctx.destination);
  decodeAll(ctx);
}

function decodeAll(context: AudioContext): void {
  preloadSounds();
  for (const file of files) {
    if (buffers.has(file)) continue;
    void downloads
      .get(file)
      // decodeAudioData detaches the buffer it is given, so each decode gets a copy.
      ?.then((bytes) => context.decodeAudioData(bytes.slice(0)))
      .then((buffer) => buffers.set(file, buffer))
      .catch(() => undefined);
  }
}

export function setSoundEnabled(on: boolean): void {
  enabled = on;
  if (on) ensureAudio();
}

export function play(name: SoundName): void {
  if (!enabled || !ctx || !master) return;
  const buffer = buffers.get(SOUNDS[name]);
  if (!buffer) return;
  try {
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(master);
    source.start();
  } catch {
    /* a device that refuses a voice must not stall the board */
  }
}
