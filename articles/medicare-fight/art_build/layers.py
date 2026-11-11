"""Cut out the podium group (left speaker + mics + sign) and the right speaker; rebuild concrete + yellow circle behind."""
import numpy as np, cv2
from PIL import Image
from rembg import remove, new_session
from scipy import ndimage
im = Image.open('still_1920.png').convert('RGB'); W, H = im.size; a = np.array(im).astype(np.float32)
s = new_session('isnet-general-use')
def cut(box):
    x0, y0, x1, y1 = box
    m = np.array(remove(im.crop(box), session=s, only_mask=True)).astype(np.float32) / 255
    lab, n = ndimage.label(m > 0.5)
    if n > 1: m *= ndimage.binary_dilation(lab == 1 + int(np.argmax(ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)))), iterations=3)
    full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = m
    return np.maximum(full, ndimage.gaussian_filter(ndimage.binary_fill_holes(full > 0.5).astype(np.float32), 0.8))
left = cut((250, 50, 1160, 1020))
right = cut((1080, 140, 1540, 940)) * (1 - (left > 0.5))
np.savez_compressed('pieces.npz', left=left, right=right)
# plate: concrete tiled (mirrored) from the clean left margin; yellow circle redrawn
strip = a[:, 0:250]; tiles = []; flip = False
while sum(t.shape[1] for t in tiles) < W: tiles.append(strip[:, ::-1] if flip else strip); flip = not flip
paper = np.concatenate(tiles, 1)[:, :W]
R, G, B = a[..., 0], a[..., 1], a[..., 2]
yel = (R > 200) & (G > 170) & (B < 140) & (left < 0.1) & (right < 0.1)
ys, xs = np.nonzero(yel); pts = np.column_stack([xs, ys]).astype(np.float32)
(cx, cy), r = cv2.minEnclosingCircle(pts[(xs > 400) & (xs < 1550)])
ycol = np.median(a[yel], 0)
circ = np.zeros((H, W), np.float32); cv2.circle(circ, (int(cx), int(cy)), int(r), 1.0, -1, cv2.LINE_AA)
tex = paper - cv2.GaussianBlur(paper, (0, 0), 3)            # keep the concrete speckle on the yellow too
plate = paper * (1 - circ[..., None]) + (ycol + tex * 0.9) * circ[..., None]
hole = ndimage.gaussian_filter(ndimage.binary_dilation((left > 0.05) | (right > 0.05), iterations=6).astype(np.float32), 2)
bg = a * (1 - hole[..., None]) + plate * hole[..., None]
Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).save('bg.png')
np.save('circle.npy', np.array([cx, cy, r]))
print('circle', round(cx), round(cy), round(r))
v = a * 0.35
for m, c in ((left, (255, 0, 0)), (right, (0, 200, 0))): v = v * (1 - m[..., None]) + (a * 0.5 + np.array(c) * 0.5) * m[..., None]
o = Image.new('RGB', (1920, 540)); o.paste(Image.fromarray(v.astype(np.uint8)).resize((960, 540)), (0, 0)); o.paste(Image.open('bg.png').resize((960, 540)), (960, 0))
o.save('/tmp/claude-0/-home-user-article-motion-template/fbb3ffda-e686-5ff7-bf3e-cd2fa0bae43e/scratchpad/mf_seg.png')
