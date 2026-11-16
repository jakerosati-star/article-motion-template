"""Masks for the White Nationalists art (1920x1080):
  jones: head and shirt from the rembg matte (clean hair and beard edges); his dark jacket, which rembg drops, from
         the jacket being exactly grayscale and never pure black, unlike the pure-black (0) crowd silhouettes.
  crowd: the pure-black silhouettes in the lower left (not Jones).
Writes masks.npz."""
import numpy as np, cv2
from PIL import Image
from scipy import ndimage
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(int)
H, W = a.shape[:2]; yy, xx = np.mgrid[0:H, 0:W]
lum = a.mean(2)
gray = (np.abs(a[..., 0] - a[..., 1]) <= 4) & (np.abs(a[..., 1] - a[..., 2]) <= 4)
from rembg import remove, new_session
rb = np.array(remove(Image.open('still_1920.png').convert('RGB'), session=new_session('isnet-general-use'), only_mask=True)) > 127
n, lab, st, _ = cv2.connectedComponentsWithStats(rb.astype(np.uint8), 8)
head = lab == lab[450, 1500]
jk = (gray & (lum > 6) & (yy > 640) & (xx > 700)).astype(np.uint8)
jk = cv2.morphologyEx(jk, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(jk, 8)
jk = np.isin(lab, [i for i in range(1, n) if st[i, 4] > 20000])
j = (head & ~((xx < 1268) & (yy < 240))) | jk          # (drop a bit of the logo touching his head)
j = ndimage.binary_fill_holes(cv2.morphologyEx(j.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)))
# the jacket's dark anti-aliased rim lies just outside the grayscale test: grow the jacket part a few px, but only
# onto pixels darker than the banner (so it takes the rim, not the backdrop)
rim = cv2.dilate(j.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))).astype(bool) & (yy > 600) & (lum < 120)
j = j | rim
crowd = (lum < 6) & ~j & (yy > 560) & (xx < 1250)
crowd = cv2.morphologyEx(crowd.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
n, lab, st, _ = cv2.connectedComponentsWithStats(crowd.astype(np.uint8), 8)
crowd = np.isin(lab, [i for i in range(1, n) if st[i, 4] > 2000])
f = lambda b: cv2.GaussianBlur(b.astype(np.float32), (0, 0), 0.7)
np.savez('masks.npz', jones=f(j).astype(np.float16), crowd=f(crowd).astype(np.float16))
