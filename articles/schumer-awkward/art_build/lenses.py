"""Lens masks for Schumer's glasses: inside the dark frames, found by flood-filling from outside the frames."""
import numpy as np, cv2
from PIL import Image
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
X0, Y0, X1, Y1 = 990, 365, 1320, 505
lum = a[Y0:Y1, X0:X1].mean(2)
frame = (lum < 45).astype(np.uint8)
frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(1 - frame, 4)
out = np.zeros(a.shape[:2], np.float32)
for i in range(1, n):
    x, y, w, h, area = st[i]
    if x > 0 and y > 0 and x + w < X1 - X0 and y + h < Y1 - Y0 and area > 3000:   # enclosed, lens-sized
        print('lens', X0 + x, Y0 + y, w, h, area)
        out[Y0:Y1, X0:X1][lab == i] = 1
out = cv2.dilate(out, np.ones((3, 3), np.uint8))
np.save('lenses.npy', out.astype(np.float16))
