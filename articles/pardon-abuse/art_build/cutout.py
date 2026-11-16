"""Foreground matte for the Pardon Abuse art: the people, Trump, his hand and the pardon are pure grayscale
(R=G=B); the paper background is warm. Thin grayscale things in the background (the printed text, the white
squiggle) are removed by a morphological open, keeping only big pieces. Writes fg.npy (0-1)."""
import numpy as np, cv2
from PIL import Image
from scipy import ndimage
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(int)
g = ((np.abs(a[..., 0] - a[..., 1]) <= 3) & (np.abs(a[..., 1] - a[..., 2]) <= 3)).astype(np.uint8)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
op = cv2.morphologyEx(g, cv2.MORPH_OPEN, k)
n, lab, st, _ = cv2.connectedComponentsWithStats(op, 8)
big = np.isin(lab, [i for i in range(1, n) if st[i, 4] > 30000]).astype(np.uint8)
fg = g & cv2.dilate(big, np.ones((7, 7), np.uint8))           # grow back to the true (fine) grayscale edge
fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
fg = ndimage.binary_fill_holes(fg).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(fg, 8)
fg = np.isin(lab, [i for i in range(1, n) if st[i, 4] > 30000]).astype(np.float32)
np.save('fg.npy', cv2.GaussianBlur(fg, (0, 0), 0.7).astype(np.float16))
