# Article motion card template

Looping social video for an article: animated art on top (or beside), headline panel below, in four sizes.

| Format | Size | Layout |
|---|---|---|
| `1x1`  | 1080×1080 | Art on top, headline panel below |
| `4x5`  | 1080×1350 | Same, larger type |
| `9x16` | 1080×1920 | Art and text centred together inside the Reels/TikTok safe zone (y 250–1600) |
| `16x9` | 1920×1080 | Headline column on the left, art on the right |

## Making a new one

There are three steps. Step 1 is custom work for each image, step 2 is fully automatic, and step 3 delivers the finals.

### 1. Make the art loop (a seamless 16:9 video, 10s works well)

**If the article art is already animated** (GIF or animated WebP), convert it with `python3 gif_to_art_loop.py <art.gif> <art_loop.mp4>`. It keeps the frame timings exactly and repeats whole cycles to get close to 10s. Upload animated art to the repo rather than attaching it in chat, since chat attachments arrive as a single still frame.

**If the art is a still**, animate it yourself:

`example/art_build/` shows how the Great Wave art was done:

1. **`cutouts.py`**: cuts the elements that should move in and out (blimp, cat, stickers) out of the still, using `rembg`. If the cutout misses part of an element, as the cat's hat was, add a hand-drawn polygon to the mask.
2. **`animate_art.py`**:
   - Fills in the background behind the cutouts (inpainting, plus mirrored texture for big areas like the cat).
   - Composites each frame:
     - Elements fly or pop in, hold, then leave before the loop point.
     - Extra elements are tossed through or pop on and off mid-loop.
     - Ambient motion: confetti, glow pulses, ray shimmer, camera push-in.
   - Every ambient motion uses frequencies that repeat exactly over the loop length (`wv(k, t)`), so the last frame flows into the first.

   Run it from inside `art_build/`. It writes `../art_loop.mp4`.

When reusing this for a new image, the per-image parts are the bounding boxes, the glow positions (laser, eyes, ray centre) and the entry/exit timings.

**Never start on an empty frame.** The first frame of the art loop is also the first thing people see (and the thumbnail when autoplay is off), so the main subject must be on screen at frame 0. Only secondary elements should animate in and out around the loop point.

Timing convention (10s loop):
- 0–3s: art elements enter.
- 3.5–8.5s: extra motion.
- 8.4–9.9s: elements exit.
- 10s: the scene is back to the frame-0 state.

### 2. Render the cards

Write an `article.json` next to your art loop and avatar (copy `example/article.json`):

```json
{
  "art_loop": "art_loop.mp4",
  "tag": "THE BULWARK",
  "headline": "Emergency Triad: MAGA Drowns Under a Blue Wave",
  "subhead": "Pop the champagne. Preserve the documents.",
  "authors": [{"name": "JONATHAN V. LAST"}],
  "date": "NOVEMBER 4",
  "accent": "#E4002B",
  "output_prefix": "out/blue-wave",
  "formats": ["1x1", "4x5", "9x16", "16x9"]
}
```

```bash
python3 render_card.py path/to/article.json
```

List one or more authors. With two or more, the headshots overlap and pop in one after another, and the names are joined with "AND":

```json
"authors": [{"name": "WILLIAM KRISTOL"}, {"name": "ANDREW EGGER"}]
```

**Optional fields:**
- `"date"`: leave it out (e.g. for promos) and the author name sits centered next to the photo.
- `"cta"`: for promos, adds a red call-to-action button (e.g. `"FIGHT THE GOOD FIGHT"`) at the right end of the byline row. It pops in after the byline and pulses gently. In 1:1, 4:5 and 9:16 the logo moves up to the right end of the tag row to make room.

## Author headshots

Headshots live in `headshots/` and are matched to authors by name automatically:

| Author name | File |
|---|---|
| JONATHAN V. LAST | `headshots/jonathan-v-last.jpg` |
| WILLIAM KRISTOL | `headshots/william-kristol.jpg` |

The preferred file name is the name in lowercase with hyphens (`sam-stein.jpg`), but matching is forgiving: capitals, spaces, underscores and stray spaces are ignored (`Sam Stein .JPG` works). If there's no exact match, first and last name are enough, so a byline of "JONATHAN V. LAST" would also find `jonathan-last.jpg`. `.jpg`, `.png` and `.webp` all work. Square photos work best, since they're center-cropped to a circle.

**Adding a new author:** on GitHub, open the `headshots` folder, click **Add file → Upload files**, drop in the photo (named with the author's name), and commit.

To use a one-off photo instead, add `"avatar": "photo.jpg"` to that author (the path is relative to the article folder). If no photo is found, the card renders without one and prints a warning.

Renders all four formats in about a minute. Add `--formats 1x1,9x16` to render only some. Each format also writes a `_still.png` frame from the 5-second mark for previews or thumbnails.

### 3. Deliver finals to Google Drive

Upload the four finished cards to the shared finals folder, in a subfolder for the article:

- **Finals folder:** https://drive.google.com/drive/folders/16tN90cEpdch5aAEYp4m6WMiTM4F8OEBr (folder ID `16tN90cEpdch5aAEYp4m6WMiTM4F8OEBr`)
- **Subfolder:** one per article. Match how the existing subfolders are named, and reuse the article's subfolder if it already exists rather than creating a duplicate.
- **Files:** the four `out/*_1x1.mp4`, `_4x5.mp4`, `_9x16.mp4` and `_16x9.mp4` cards. If cards are re-rendered after a revision, replace the old files in the same subfolder.

Needs the Google Drive connector to be connected for the session.

## Headline panel animation (built into `render_card.py`)

The headline comes in first and leaves last, so it stays on screen as long as possible:

| Time | Event |
|---|---|
| 0s | Headline starts rising word by word from behind a mask, already moving on the first frame |
| 0.25s | Accent seam line wipes across (vertical on the art's left edge in 16:9) |
| 0.35s | Tag box opens, then the tag text rises |
| 0.7s | Subhead fades up |
| 0.9s | Avatar(s) pop in, then name and date |
| last ~1.5s | Byline, subhead, tag and seam leave in reverse order |
| last ~0.6s | Headline sinks out, ready to loop |

Outro times are anchored to the end of the art loop, so loops longer or shorter than 10s still work.

## Branding

The red tag always reads **THE BULWARK** (the default when an article's `tag` is left out). For paid media we don't break out the show or newsletter name.


Every card carries a white Bulwark logo at 70% opacity, always on screen (from `BW Logos (transparent)/`):

| Format | Lockup | Placement |
|---|---|---|
| 1:1, 4:5 | Horizontal | Right end of the byline row |
| 9:16 | Stacked | Right end of the byline row |
| 16:9 | Horizontal | Bottom of the text column |

Sizes, positions and opacity are in the `LOGOS` table in `render_card.py`. Add `"logo": false` to an article's settings to leave it off.

## Type

Special Gothic variable font (`fonts/`):

| Element | Weight | Width |
|---|---|---|
| Headline, tag | 700 | 75 (condensed) |
| Subhead | 600 | 100 |
| Byline | 700 | 90 |

Short headlines that fit on one line are enlarged, up to 1.5×. When the text block leaves more than 60px of spare room in the panel, it is moved down into that room. In 1:1 and 4:5, 30% of the spare room goes above the text and 70% below, so the headline stays close to the art (the `bias` setting in `LAYOUTS`). Headlines wrap normally. If that would leave one word alone on the last line, the lines are evened out instead. Sizes and positions per format are in the `LAYOUTS` table at the top of `render_card.py`.

## Requirements

`ffmpeg`, Python 3 with `pillow` and `numpy`. The art step also needs `opencv-python-headless` and `rembg[cpu]`.
