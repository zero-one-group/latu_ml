# Latu ML identity assets

The companion package keeps Latu's identity and flips one thing: the avatar tile is the
negative of Latu's — ember ground, ink letter — so the two packages are told apart at a glance
in a list and still read as one family. The wordmark is "Latu ML", with "ML" set in the
Javanese colour (#6E4A7E) as a descriptor; ꦭꦠꦸ stays the name. Everything else — palette,
fonts, the shared baseline, the x-height rule, the tile alignment, and the rule that the script
appears only as outlined SVG — is Latu's, recorded in that repo's `assets/README.md`.

| File | Use |
|---|---|
| `latu-ml-lockup.svg` / `-dark.svg` (+ `@2x.png`) | **Primary lockup**, `Latu ML \| ꦭꦠꦸ`, light / dark grounds |
| `latu-ml-lockup-with-mark*.svg` | Tile + lockup, for a docs header or social card |
| `latu-ml-lockup-stacked*.svg` | Latin over Javanese, for banners |
| `latu-ml-avatar.svg`, `latu-ml-avatar-{512,256,128}.png` | GitHub avatar, hex.pm |
| `favicon.svg`, `favicon.ico`, `favicon-{16,32,48}.png` | Favicon set (the tile) |

The README's image URLs are relative while the repo is private. When it goes public on GitHub,
make them absolute (`raw.githubusercontent.com/zero-one-group/latu_ml/main/assets/...`), as
Latu's are: hex.pm renders the README without `assets/`, which is deliberately not in the Hex
package.

Drawn by the same generator as Latu's assets (`build_ml.py` in the design-session pack, not in
the repo).
