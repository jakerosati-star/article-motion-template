"""Pens: traced as capsules along each pen's axis, refined by colour. Ink: dark/teal marker strokes, rivers removed."""
import numpy as np, cv2, json
from PIL import Image
from scipy import ndimage
a = np.array(Image.open('still_1920.png')).astype(np.float32); H, W = a.shape[:2]
R, G, B = a[..., 0], a[..., 1], a[..., 2]; sat = a.max(2) - a.min(2); lum = a.mean(2)
# tip (x, y) -> far end (x, y), half-width
PENS = {'A': ((499, 435), (-60, -40), 58), 'B': ((838, 154), (965, -30), 52), 'C': ((1295, 109), (1400, -30), 50),
        'D': ((1414, 474), (1960, 50), 58), 'E': ((1626, 845), (1960, 440), 56), 'F': ((909, 896), (745, 1100), 54)}
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
masks = {}
for k, ((tx, ty), (ex, ey), hw) in PENS.items():
    d = np.array([ex - tx, ey - ty], np.float32); L = np.linalg.norm(d); u = d / L
    along = (xx - tx) * u[0] + (yy - ty) * u[1]; across = np.abs((xx - tx) * -u[1] + (yy - ty) * u[0])
    # cone near the tip widens from 0 to the full body over the first 90px
    allowed = (along > -10) & (across < np.clip((along + 10) * 0.75, 0, hw + 14))
    if k in ('C', 'F'):   # light-blue pens over land: anything bluer than the map
        col = ((B > R + 18) & (lum > 95)) | ((lum < 90) & (along < 80))
    else:
        col = ((sat > 85) & (B > R + 50)) | ((lum < 90) & (along < 80))  # pen plastic, or the dark felt tip
    m = ndimage.binary_closing(allowed & col, iterations=4)
    m = ndimage.binary_fill_holes(m) & allowed
    lab, n = ndimage.label(m)
    if n: m = lab == 1 + int(np.argmax(ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))))
    masks[k] = ndimage.gaussian_filter(m.astype(np.float32), 1.0)
pen_any = np.max(np.stack(list(masks.values())), 0)
np.savez_compressed('pens.npz', **masks)
ink = ((B - (R + G) / 2) > 22) & (lum < 160) & (pen_any < 0.05)
ink = ndimage.binary_opening(ink, iterations=3)                          # drops thin rivers and text
ink = ndimage.binary_dilation(ink, iterations=2)
v = a.copy()
for k, m in masks.items(): v = v * (1 - m[..., None]) + np.array([255, 0, 255]) * m[..., None]
v[ink] = [0, 255, 0]
np.save('ink.npy', ndimage.gaussian_filter(ink.astype(np.float32), 1.0))
# fit an ellipse to each dark circle's ink, near where it sits
SEEDS = {'B': (800, 205, 230, 120), 'A': (650, 490, 300, 140), 'D': (1235, 520, 290, 140), 'E': (1650, 890, 220, 230)}
ell = {}
for k, (cx, cy, rx, ry) in SEEDS.items():
    box = ink.copy(); box[:max(0, cy - ry)] = 0; box[cy + ry:] = 0; box[:, :max(0, cx - rx)] = 0; box[:, cx + rx:] = 0
    pts = np.column_stack(np.nonzero(box))[:, ::-1].astype(np.float32)
    (ex, ey), (w, h), ang = cv2.fitEllipse(pts); ell[k] = [ex, ey, w / 2, h / 2, ang]
json.dump(ell, open('ellipses.json', 'w'))
for k, (ex, ey, rx, ry, ang) in ell.items():
    cv2.ellipse(v, (int(ex), int(ey)), (int(rx), int(ry)), ang, 0, 360, (255, 255, 0), 3)
Image.fromarray(v.astype(np.uint8)).resize((1200, 675)).save('/tmp/claude-0/-home-user-article-motion-template/fbb3ffda-e686-5ff7-bf3e-cd2fa0bae43e/scratchpad/dp_seg.png')
