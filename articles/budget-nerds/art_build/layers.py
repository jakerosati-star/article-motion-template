"""Cut out Vought, the dome and the bill stack (rembg on crops); rebuild the green panel + paper behind them."""
import numpy as np
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
    full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = m; return full
vought = cut((840, 50, 1180, 1040))
dome = cut((1100, 100, 1700, 960)) * (1 - (vought > 0.3))
bills = cut((300, 80, 1110, 1080)) * (1 - (vought > 0.3)) * (1 - (dome > 0.3))
for k, m in (('vought', vought), ('dome', dome), ('bills', bills)):
    m[:] = np.maximum(m, ndimage.gaussian_filter(ndimage.binary_fill_holes(m > 0.5).astype(np.float32), 0.8))
np.savez_compressed('pieces.npz', vought=vought, dome=dome, bills=bills)
# plate: paper texture (mirrored from the clean left margin) + flat green panel
strip = a[:, 0:300]; tiles = []; flip = False
while sum(t.shape[1] for t in tiles) < W: tiles.append(strip[:, ::-1] if flip else strip); flip = not flip
paper = np.concatenate(tiles, 1)[:, :W]
G0 = (308, 207, 1610, 872)
gcol = np.median(a[300:600, 320:380].reshape(-1, 3), 0)
plate = paper.copy(); plate[G0[1]:G0[3], G0[0]:G0[2]] = gcol + (paper[G0[1]:G0[3], G0[0]:G0[2]] - paper.mean((0, 1))) * 0.25
occ = np.maximum(np.maximum(vought, dome), bills)
hole = ndimage.gaussian_filter(ndimage.binary_dilation(occ > 0.05, iterations=8).astype(np.float32), 2)
bg = a * (1 - hole[..., None]) + plate * hole[..., None]
Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).save('bg.png')
v = a * 0.3
for m, c in ((vought, (255, 0, 0)), (dome, (0, 0, 255)), (bills, (255, 200, 0))): v = v * (1 - m[..., None]) + (a * 0.5 + np.array(c) * 0.5) * m[..., None]
s2 = Image.new('RGB', (1920, 540)); s2.paste(Image.fromarray(v.astype(np.uint8)).resize((960, 540)), (0, 0)); s2.paste(Image.open('bg.png').resize((960, 540)), (960, 0))
s2.save('/tmp/claude-0/-home-user-article-motion-template/fbb3ffda-e686-5ff7-bf3e-cd2fa0bae43e/scratchpad/bn_seg.png')
