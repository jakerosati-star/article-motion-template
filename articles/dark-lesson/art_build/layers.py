"""Plane line art and bullet-hole dots (white on fire) separated from the flames; fire plate with the white removed."""
import numpy as np, cv2
from PIL import Image
from scipy import ndimage
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32); H, W = a.shape[:2]
IN = (108, 91, 1811, 989)
inner = np.zeros((H, W), bool); inner[IN[1]:IN[3], IN[0]:IN[2]] = True
sat = a.max(2) - a.min(2)
white = (a.min(2) > 175) & (sat < 70) & inner
lab, n = ndimage.label(white)
dots = np.zeros((H, W), bool); dot_list = []
for sl, idx in zip(ndimage.find_objects(lab), range(1, n + 1)):
    comp = lab[sl] == idx; h, w = comp.shape; area = comp.sum()
    if 60 < area < 600 and 0.6 < h / max(w, 1) < 1.6 and area / (h * w) > 0.55:
        dots[sl] |= comp; dot_list.append(((sl[1].start + sl[1].stop) / 2, (sl[0].start + sl[0].stop) / 2, max(h, w) / 2))
lines = white & ~dots
soft = lambda m, s: ndimage.gaussian_filter(ndimage.binary_dilation(m, iterations=1).astype(np.float32), s)
np.savez_compressed('pieces.npz', lines=soft(lines, 0.6), dots=soft(dots, 0.6), inner=inner.astype(np.float32))
np.save('dots.npy', np.array(dot_list, np.float32))
hole = cv2.dilate(white.astype(np.uint8), np.ones((7, 7), np.uint8))
fire = cv2.inpaint(a.astype(np.uint8), hole, 7, cv2.INPAINT_TELEA).astype(np.float32)
Image.fromarray(fire.astype(np.uint8)).save('fire.png')
print('dots', len(dot_list))
v = a * 0.4; v[lines] = [0, 255, 255]; v[dots] = [255, 0, 255]
s = Image.new('RGB', (1920, 540)); s.paste(Image.fromarray(v.astype(np.uint8)).resize((960, 540)), (0, 0)); s.paste(Image.open('fire.png').resize((960, 540)), (960, 0))
s.save('/tmp/claude-0/-home-user-article-motion-template/fbb3ffda-e686-5ff7-bf3e-cd2fa0bae43e/scratchpad/dl_seg.png')
