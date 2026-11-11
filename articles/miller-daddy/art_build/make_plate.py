"""Clean spotlit wall behind the two figures: smooth lighting filled in from the visible wall + concrete texture
tiled from the clean band at the right."""
import numpy as np, cv2
from PIL import Image
from scipy import ndimage
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32); H, W = a.shape[:2]
m = np.load('rembg_all.npy'); fig = ndimage.binary_dilation(m > 0.05, iterations=8)
w = (~fig).astype(np.float32); light = a.copy()
for sig in (40, 80, 160):
    num = cv2.GaussianBlur(a * w[..., None], (0, 0), sig); den = cv2.GaussianBlur(w, (0, 0), sig)[..., None]
    light = np.where(fig[..., None], num / np.maximum(den, 1e-4), light)
light = cv2.GaussianBlur(light, (0, 0), 25)
hp = a - cv2.GaussianBlur(a, (0, 0), 6)
src = hp[:, 1500:1900]; tiles = []; flip = False
while sum(t.shape[1] for t in tiles) < W: tiles.append(src[:, ::-1] if flip else src); flip = not flip
tex = np.concatenate(tiles, 1)[:, :W]
plate = np.where(fig[..., None], light + tex, a)
soft = ndimage.gaussian_filter(fig.astype(np.float32), 3)[..., None]
plate = a * (1 - soft) + plate * soft
Image.fromarray(np.clip(plate, 0, 255).astype(np.uint8)).save('bg.png')
