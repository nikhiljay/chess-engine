# Piece sets

## cburnett/ — lichess's default pieces

The twelve SVGs in `cburnett/` are taken from `public/piece/cburnett/` in
[lichess-org/lila](https://github.com/lichess-org/lila). They are the set lichess
draws by default.

**Licence: GPLv2+**, by [Colin M.L. Burnett](https://en.wikipedia.org/wiki/User:Cburnett) —
lila's `COPYING.md` lists them under *Exceptions (free)*. Unlike the sounds in
`../sounds/standard/`, these are ours to redistribute, with attribution.

`scripts/pieces.ts` bakes them into `src/pieces.ts` as base64 data URIs, so the
board makes no extra requests and no piece appears late. Regenerate with:

    npm run pieces

The board itself is drawn with CSS colours (`#f0d9b5` / `#b58863` in `index.html`),
not with lichess's `brown.png`: their board images fall under "the other artwork in
public/images", which `COPYING.md` lists as *non-free*.

Refresh the files with:

    base=https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett
    for p in wK wQ wR wB wN wP bK bQ bR bB bN bP; do curl -sO "$base/$p.svg"; done
