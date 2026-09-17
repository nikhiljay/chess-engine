# Sound samples

Both sets are lichess's, taken from `public/sound/` in
[lichess-org/lila](https://github.com/lichess-org/lila). The active set is chosen
by `THEME` in `src/sound-theme.ts`; `scripts/build.ts` copies only that set into
`dist/sounds/`.

## standard/ — lichess's default set (currently active)

`Move.mp3`, `Capture.mp3`, `GenericNotify.mp3` (played at the end of a game) and
`Silence.mp3` (lichess's default set has no check sound: `standard/Check.mp3` is a
symlink to `Silence.mp3`, and `Victory`, `Defeat` and `Draw` all symlink to
`GenericNotify.mp3`).

**Licence: none stated.** lila's `COPYING.md` lists "the other sounds in
public/sound" under *Exceptions (non-free)*, with no author and no licence — so
these are not ours to redistribute. They are fine on a private page; swap to `sfx`
before sharing this artifact with anyone.

## sfx/ — the free alternative

`Move.mp3`, `Capture.mp3`, `Check.mp3`, `Victory.mp3`, `Defeat.mp3`, `Draw.mp3`,
by [Enigmahack](https://github.com/Enigmahack), **AGPLv3+** per lila's `COPYING.md`.
Freely redistributable with attribution. Brighter and more synthetic than the
default set, and it has a real check sound.

Refresh either set with:

    base=https://raw.githubusercontent.com/lichess-org/lila/master/public/sound
    curl -sSO "$base/standard/Move.mp3"   # etc.
