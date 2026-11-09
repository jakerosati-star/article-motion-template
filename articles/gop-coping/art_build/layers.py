"""Balloon and string cutouts by colour; blue paper (and the suit under the string) rebuilt behind them."""
import numpy as np, cv2
from PIL import Image
from scipy import ndimage
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32); H, W = a.shape[:2]
R, G, B = a[..., 0], a[..., 1], a[..., 2]
box = np.zeros((H, W), bool); box[30:1010, 320:780] = True
col = ((R - B > 25) & (R - G > 25)) | (a.min(2) > 200)
m = ndimage.binary_closing(col & box, iterations=2)
balloon = ndimage.binary_fill_holes(ndimage.binary_closing(m & (np.mgrid[0:H, 0:W][0] < 600), iterations=4))
lab, n = ndimage.label(balloon); balloon = lab == 1 + int(np.argmax(ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))))
string = m & ~balloon & (np.mgrid[0:H, 0:W][0] >= 560)
lab, n = ndimage.label(ndimage.binary_dilation(string, iterations=2))
sz = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)); string = string & np.isin(lab, 1 + np.where(sz > 300)[0])
string = ndimage.binary_dilation(string, iterations=1)
np.savez_compressed('pieces.npz', balloon=ndimage.gaussian_filter(balloon.astype(np.float32), 1.0),
                    string=ndimage.gaussian_filter(string.astype(np.float32), 0.7))
# plate: blue paper tiled (mirrored) from the clean left margin; suit under the string by inpaint
strip = a[:, 0:300]; tiles = []; flip = False
while sum(t.shape[1] for t in tiles) < W: tiles.append(strip[:, ::-1] if flip else strip); flip = not flip
paper = np.concatenate(tiles, 1)[:, :W]
hole = cv2.dilate((balloon | string).astype(np.uint8), np.ones((11, 11), np.uint8))
blue = (B > R + 20) | (balloon > 0)
inp = cv2.inpaint(a.astype(np.uint8), hole, 9, cv2.INPAINT_TELEA).astype(np.float32)
blue_near = cv2.dilate(blue.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
fill = np.where((blue_near & ~((inp.mean(2) < 90)))[..., None], paper, inp)
soft = ndimage.gaussian_filter(hole.astype(np.float32), 2)[..., None]
bg = a * (1 - soft) + fill * soft
Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).save('bg.png')
v = a.copy(); v[balloon] = [0, 255, 0]; v[string] = [255, 0, 255]
s = Image.new('RGB', (960, 540)); s.paste(Image.fromarray(v.astype(np.uint8)).crop((240, 0, 1080, 1080)).resize((420, 540)), (0, 0))
s.paste(Image.open('bg.png').crop((240, 0, 1080, 1080)).resize((420, 540)), (440, 0))
s.save('/tmp/claude-0/-home-user-article-motion-template/fbb3ffda-e686-5ff7-bf3e-cd2fa0bae43e/scratchpad/gc_seg.png')
