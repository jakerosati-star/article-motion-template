"""Schumer vs. Van Hollen art: the two photo cards seesaw: Schumer slides up while Van Hollen's card slides down,
then they swap, home again at the loop point (frame 0 = the art), like a balance tipping between them. A glint sweeps across Schumer's glasses (0.45s, again at 5.6s) and the clouds behind
Van Hollen drift. The background behind the cards is rebuilt by inpainting (only a thin band of it is ever seen).
Run from art_build/ (after lenses.py and cards.py). Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
art = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
cards = np.load('cards.npz')
SCH = cards['schumer'].astype(np.float32); VH = cards['vh'].astype(np.float32)

# --- background without the cards (inpainted at half size, used only where a card was)
hole = cv2.dilate(((SCH + VH) > 0.02).astype(np.uint8), np.ones((7, 7), np.uint8)) * 255
small = cv2.resize(art.astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_AREA)
plate_s = cv2.inpaint(small, cv2.resize(hole, (W // 2, H // 2)), 6, cv2.INPAINT_TELEA)
plate = np.where((hole > 0)[..., None], cv2.resize(plate_s, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32), art)
# around Schumer below the clouds, the Capitol is all vertical columns: extend each pixel column straight up/down from
# the nearest visible pixel instead (inpainting smears them)
def vertical_fill(img, holemask):
    hm = holemask; idx = np.arange(H)[:, None].repeat(W, 1)
    up_i = np.where(~hm, idx, -1); up_i = np.maximum.accumulate(up_i, axis=0)
    dn_i = np.where(~hm, idx, H); dn_i = np.minimum.accumulate(dn_i[::-1], axis=0)[::-1]
    use_up = (idx - up_i) <= (dn_i - idx)
    src = np.where(use_up, up_i, dn_i); src = np.clip(src, 0, H - 1)
    ok = (up_i >= 0) | (dn_i < H)
    out = img.copy(); cols = np.arange(W)[None, :].repeat(H, 0)
    out[hm & ok] = img[src, cols][hm & ok]
    return out
sch_hole = cv2.dilate((SCH > 0.02).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
cap_zone = np.zeros((H, W), bool); cap_zone[300:, :1300] = True              # Capitol, not clouds
vf = vertical_fill(art, sch_hole)
plate = np.where((sch_hole & cap_zone)[..., None], vf, plate)

# --- sky in Van Hollen's panel: the panel minus his (pure grayscale) photo; clouds continue behind him
PX0, PY0, PX1, PY1 = 1308, 150, 1814, 880
gray = (np.abs(art[..., 0] - art[..., 1]) <= 6) & (np.abs(art[..., 1] - art[..., 2]) <= 6)
sky = np.zeros((H, W), np.uint8); sky[PY0:PY1, PX0:PX1] = 1; sky[gray] = 0
sky = cv2.morphologyEx(sky, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(1 - sky[PY0:PY1, PX0:PX1], 4)
for i in range(1, n):
    if st[i, 4] < 3000: sky[PY0:PY1, PX0:PX1][lab == i] = 1
skyf = cv2.GaussianBlur(sky.astype(np.float32), (0, 0), 0.8) * sky.astype(np.float32)
crop = art[PY0:PY1, PX0:PX1].astype(np.uint8)
shole = cv2.dilate((1 - sky[PY0:PY1, PX0:PX1]).astype(np.uint8), np.ones((9, 9), np.uint8)) * 255
sky_src = cv2.inpaint(crop, shole, 9, cv2.INPAINT_TELEA).astype(np.float32)
cy_, cx_ = np.mgrid[0:PY1 - PY0, 0:PX1 - PX0].astype(np.float32)
DRIFT = 40.0

# --- glasses glint
lens = cv2.GaussianBlur(np.load('lenses.npy').astype(np.float32), (0, 0), 1.0)
GLINTS = [0.45, 5.6]; SWEEP = 0.5
def glint(t):
    g = np.zeros((H, W), np.float32)
    for t0 in GLINTS:
        for lx0, lx1, delay in ((1000, 1135, 0.0), (1170, 1305, 0.07)):
            s = (t - t0 - delay) / SWEEP
            if not 0 <= s <= 1: continue
            s = 0.5 - 0.5 * math.cos(math.pi * s)
            pos = lx0 - 60 + (lx1 - lx0 + 120) * s
            d = (xx - pos) + 0.55 * (yy - 430)
            band = np.exp(-(d / 11) ** 2) + 0.7 * np.exp(-((d + 30) / 4) ** 2) + 0.18
            g = np.maximum(g, band * ((xx >= lx0 - 10) & (xx <= lx1 + 10)) * math.sin(math.pi * s) ** 0.6)
    return np.clip(g * lens, 0, 1)

# --- seesaw: (dx, dy) in px at the peak (2.5s); reversed at 7.5s
SCH_MOVE = (0.0, -16.0)
VH_MOVE = (0.0, 16.0)
def shift(img, dx, dy):
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H), flags=cv2.INTER_LINEAR)

def frame(t):
    f = art.copy()
    dx = DRIFT * (0.5 - 0.5 * math.cos(TAU * t / P))
    moved = cv2.remap(sky_src, cx_ - dx, cy_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    a = skyf[PY0:PY1, PX0:PX1, None]
    f[PY0:PY1, PX0:PX1] = f[PY0:PY1, PX0:PX1] * (1 - a) + moved * a
    g = glint(t)
    if g.max() > 0: f = 255 - (255 - f) * (1 - g[..., None] * 0.95)
    e = math.sin(TAU * t / P)                             # 0 at the loop point, +1 at 2.5s, -1 at 7.5s
    out = plate.copy()
    for m, (mx, my) in ((SCH, SCH_MOVE), (VH, VH_MOVE)):  # premultiplied layers: no fringes
        al = shift(m, mx * e, my * e)[..., None]
        out = out * (1 - al) + shift(f * m[..., None], mx * e, my * e)
    return out

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        for t in map(float, sys.argv[1:]): Image.fromarray(np.clip(frame(t), 0, 255).astype(np.uint8)).save(f'_t{t:.2f}.png')
        sys.exit()
    enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                           stdin=subprocess.PIPE)
    for i in range(N): enc.stdin.write(np.clip(frame(i / FPS), 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close(); enc.wait()
