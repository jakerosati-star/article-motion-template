"""Masks for the two photo "cards" (1920x1080, 0-1):
  schumer: his photo, straight right edge x=1306 and bottom y=878. Against the blue Capitol the photo is told apart
           by being pure grayscale; above y=150 his hair meets gray clouds, so there the rembg matte decides.
  vh:      the blue sky panel plus his photo, which runs below the panel to y=990 (straight left edge x=1345).
Writes cards.npz."""
import numpy as np, cv2
from PIL import Image
from scipy import ndimage
from rembg import remove, new_session
im = Image.open('still_1920.png').convert('RGB'); a = np.array(im).astype(int)
H, W = a.shape[:2]; yy, xx = np.mgrid[0:H, 0:W]
gray = ((np.abs(a[..., 0] - a[..., 1]) <= 6) & (np.abs(a[..., 1] - a[..., 2]) <= 6)).astype(np.uint8)
lum = a.mean(2)
rb = np.array(remove(im, session=new_session('isnet-general-use'), only_mask=True)) > 127

def biggest(b, seed):
    n, lab, _, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), 8)
    return lab == lab[seed[1], seed[0]]

# Schumer
box = (xx > 500) & (xx < 1306) & (yy < 878)
s = np.where(yy < 150, rb, gray.astype(bool)) & box
s = cv2.morphologyEx(s.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
s = biggest(s, (1100, 450))
core = cv2.morphologyEx(s.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
core = biggest(core, (1100, 450))                                       # thin white Capitol columns touching his shoulder drop off
s = s & cv2.dilate(core.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
s = ndimage.binary_fill_holes(cv2.morphologyEx(s.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))) & box
# Van Hollen
panel = (xx >= 1308) & (xx < 1814) & (yy >= 150) & (yy < 880)
lower = (xx >= 1345) & (xx < 1814) & (yy >= 880) & (yy < 990)
shoulder = (xx >= 1814) & (xx < 1860) & (yy >= 700) & (yy < 990) & (lum < 110)          # his suit past the panel edge
shoulder = shoulder & ndimage.binary_dilation(lower | panel, iterations=60)
v = panel | lower | ndimage.binary_opening(shoulder, iterations=2)
f = lambda b: cv2.GaussianBlur(b.astype(np.float32), (0, 0), 0.7)
np.savez('cards.npz', schumer=f(s).astype(np.float16), vh=f(v).astype(np.float16))
