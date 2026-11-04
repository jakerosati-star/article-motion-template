"""Split the AOC collage into layers: background, white band, Virginia map, foreground (AOC + voters)."""
import numpy as np, cv2
from PIL import Image
from scipy import ndimage

a = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
H, W = a.shape[:2]

# 1. background: the clean speckled strip below everything, mirrored to fill the frame
strip = a[970:1080]
tiles = []; flip = False
while sum(t.shape[0] for t in tiles) < H:
    tiles.append(strip[::-1] if flip else strip); flip = not flip
bg = np.concatenate(tiles)[:H]
Image.fromarray(bg.astype(np.uint8)).save('L_bg.png')

# 2. band: flat translucent white rectangle
BAND = (128, 334, 1792, 748)
band_rgb = np.median(a[400:700, 140:300].reshape(-1, 3), 0)
np.save('band.npy', np.array([*BAND, *band_rgb]))

# 3. map: alpha from "blueness" (blue minus red), flat colour
BLUE = np.array([113, 193, 254], np.float32)
blu = np.clip(((a[..., 2] - a[..., 0]) - 5) / 136, 0, 1)

# 4. foreground: rembg mask with holes filled, blue removed
m = np.array(Image.open('rembg_mask.png')).astype(np.float32) / 255
hard = ndimage.binary_fill_holes(m > 0.5)
core = ndimage.binary_erosion(hard, iterations=2)
fg_a = np.where(core, 1.0, m) * (1 - blu)          # solid inside, soft only at the outline
fg_a[blu > 0.5] = 0
# rembg also grabbed the map's left lobe: drop its anti-aliased edge (pale bluish pixels) from the foreground
lum = a.mean(2); near_map = ndimage.binary_dilation(blu > 0.5, iterations=2)
fg_a[near_map & ~(((a[..., 2] - a[..., 0]) < 3) & (lum < 235))] = 0
# and any leftover specks not connected to the main figure group
lab, n = ndimage.label(fg_a > 0.3)
if n > 1:
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)); keep = np.isin(lab, 1 + np.where(sizes > 2000)[0])
    fg_a[~ndimage.binary_dilation(keep, iterations=3)] = 0

# hidden part of the map (behind AOC / voters): fill inside the map's convex hull where the foreground covers it
mm = (blu > 0.5).astype(np.uint8)
cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
hull = np.zeros_like(mm); cv2.fillPoly(hull, [cv2.convexHull(np.concatenate(cnts))], 1)
cover = cv2.dilate((fg_a > 0.05).astype(np.uint8), np.ones((9, 9), np.uint8))
map_a = np.maximum(blu, (hull & cover).astype(np.float32))
map_rgba = np.dstack([np.broadcast_to(BLUE, a.shape), map_a * 255]).astype(np.uint8)
Image.fromarray(map_rgba, 'RGBA').save('L_map.png')

fg_rgba = np.dstack([a, fg_a * 255]).astype(np.uint8)
Image.fromarray(fg_rgba, 'RGBA').save('L_fg.png')

# rebuild check
def over(base, rgb, al): return base * (1 - al[..., None]) + rgb * al[..., None]
x0, y0, x1, y1 = BAND
band_a = np.zeros((H, W), np.float32); band_a[y0:y1, x0:x1] = 1
comp = over(bg, band_rgb, band_a)
comp = over(comp, BLUE, map_a)
comp = over(comp, a, fg_a)
print('mean abs diff vs original:', np.abs(comp - a).mean().round(2))
Image.fromarray(comp.astype(np.uint8)).save('rebuild.png')
