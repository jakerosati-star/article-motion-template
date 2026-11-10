"""Cut out Trump (left) and Hegseth (right) with rembg on crops; mark the water for shimmer."""
import numpy as np
from PIL import Image
from rembg import remove, new_session
from scipy import ndimage
im = Image.open('still_1920.png').convert('RGB'); W, H = im.size; a = np.array(im).astype(np.float32)
s = new_session('isnet-general-use'); out = {}
for k, (x0, y0, x1, y1) in {'trump': (0, 40, 600, 1080), 'hegseth': (1330, 40, 1920, 1080)}.items():
    m = np.array(remove(im.crop((x0, y0, x1, y1)), session=s, only_mask=True)).astype(np.float32) / 255
    lab, n = ndimage.label(m > 0.5)
    if n > 1: m *= ndimage.binary_dilation(lab == 1 + int(np.argmax(ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)))), iterations=3)
    full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = m
    full = np.maximum(full, ndimage.gaussian_filter(ndimage.binary_fill_holes(full > 0.5).astype(np.float32), 0.8))
    out[k] = full
R, G, B = a[..., 0], a[..., 1], a[..., 2]
water = (B > R + 25) & (B > 60) & (np.maximum(out['trump'], out['hegseth']) < 0.1)
out['water'] = ndimage.gaussian_filter(ndimage.binary_opening(water, iterations=2).astype(np.float32), 3)
np.savez_compressed('pieces.npz', **out)
v = a * 0.4
for k, c in (('trump', (255, 0, 0)), ('hegseth', (0, 255, 0)), ('water', (255, 255, 0))): v = v * (1 - out[k][..., None]) + (a * 0.5 + np.array(c) * 0.5) * out[k][..., None]
Image.fromarray(v.astype(np.uint8)).resize((960, 540)).save('/tmp/claude-0/-home-user-article-motion-template/fbb3ffda-e686-5ff7-bf3e-cd2fa0bae43e/scratchpad/dw_seg.png')
