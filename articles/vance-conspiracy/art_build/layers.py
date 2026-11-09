"""Cut out each collage piece by running rembg on a crop around it."""
import numpy as np
from PIL import Image
from rembg import remove, new_session
from scipy import ndimage
im = Image.open('still_1920.png').convert('RGB'); W, H = im.size
BOX = {'tucker': (70, 0, 563, 467), 'hat': (614, 0, 947, 166), 'trump': (646, 186, 1229, 870), 'vance': (1152, 0, 1843, 531),
       'thiel': (301, 614, 730, 1080), 'pepe': (806, 922, 1005, 1080), 'rubio': (1050, 762, 1376, 1080), 'donjr': (1363, 563, 1754, 1043)}
s = new_session('isnet-general-use'); out = {}
for k, (x0, y0, x1, y1) in BOX.items():
    m = np.array(remove(im.crop((x0, y0, x1, y1)), session=s, only_mask=True)).astype(np.float32) / 255
    lab, n = ndimage.label(m > 0.5)                                      # keep the biggest piece in the crop
    if n > 1: m *= ndimage.binary_dilation(lab == 1 + int(np.argmax(ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)))), iterations=3)
    full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = m; out[k] = full
np.savez_compressed('pieces.npz', **out)
a = np.array(im).astype(float); v = a * 0.3
cols = [(255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 200, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255)]
for j, (k, m) in enumerate(out.items()): v = v * (1 - m[..., None]) + (a * 0.5 + np.array(cols[j]) * 0.5) * m[..., None]
Image.fromarray(v.astype('uint8')).resize((960, 540)).save('/tmp/claude-0/-home-user-article-motion-template/fbb3ffda-e686-5ff7-bf3e-cd2fa0bae43e/scratchpad/vc_seg.png')
