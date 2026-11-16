"""Schumer vs. Van Hollen art: a glint of light sweeps across Schumer's glasses (0.45s, again at 5.6s), and the
clouds in the blue panel behind Van Hollen drift slowly right and back (seamless over the loop).
Frame 0 = the art. Run from art_build/ (after lenses.py). Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
art = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

# --- sky behind Van Hollen: the blue panel minus his (pure grayscale) photo
PX0, PY0, PX1, PY1 = 1308, 150, 1814, 880
gray = (np.abs(art[..., 0] - art[..., 1]) <= 6) & (np.abs(art[..., 1] - art[..., 2]) <= 6)
sky = np.zeros((H, W), np.uint8); sky[PY0:PY1, PX0:PX1] = 1; sky[gray] = 0
sky = cv2.morphologyEx(sky, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(1 - sky[PY0:PY1, PX0:PX1], 4)
for i in range(1, n):                                   # small enclosed white clouds belong to the sky
    if st[i, 4] < 3000: sky[PY0:PY1, PX0:PX1][lab == i] = 1
skyf = cv2.GaussianBlur(sky.astype(np.float32), (0, 0), 0.8) * sky.astype(np.float32)   # keep his edge crisp
crop = art[PY0:PY1, PX0:PX1].astype(np.uint8)
hole = cv2.dilate((1 - sky[PY0:PY1, PX0:PX1]).astype(np.uint8), np.ones((9, 9), np.uint8)) * 255
sky_src = cv2.inpaint(crop, hole, 9, cv2.INPAINT_TELEA).astype(np.float32)    # clouds continue behind him
cy, cx = np.mgrid[0:PY1 - PY0, 0:PX1 - PX0].astype(np.float32)
DRIFT = 28.0

# --- glasses glint: a soft diagonal streak (plus a thin second one) sweeping left to right, inside the lenses
lens = np.load('lenses.npy').astype(np.float32)
lens = cv2.GaussianBlur(lens, (0, 0), 1.0)
GLINTS = [0.45, 5.6]; SWEEP = 0.5
def glint(t):
    g = np.zeros((H, W), np.float32)
    for t0 in GLINTS:
        for lx0, lx1, delay in ((1000, 1135, 0.0), (1170, 1305, 0.07)):    # left lens, right lens (slightly later)
            s = (t - t0 - delay) / SWEEP
            if not 0 <= s <= 1: continue
            s = 0.5 - 0.5 * math.cos(math.pi * s)
            pos = lx0 - 60 + (lx1 - lx0 + 120) * s
            d = (xx - pos) + 0.55 * (yy - 430)                             # diagonal band, leaning right
            band = np.exp(-(d / 11) ** 2) + 0.7 * np.exp(-((d + 30) / 4) ** 2) + 0.18   # streaks + a faint whole-lens flash
            region = (xx >= lx0 - 10) & (xx <= lx1 + 10)
            g = np.maximum(g, band * region * math.sin(math.pi * s) ** 0.6)
    return g * lens

def frame(t):
    f = art.copy()
    dx = DRIFT * (0.5 - 0.5 * math.cos(TAU * t / P))
    moved = cv2.remap(sky_src, cx - dx, cy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    a = skyf[PY0:PY1, PX0:PX1, None]
    f[PY0:PY1, PX0:PX1] = f[PY0:PY1, PX0:PX1] * (1 - a) + moved * a
    g = np.clip(glint(t)[..., None], 0, 1) * 0.95
    if g.max() > 0: f = 255 - (255 - f) * (1 - g)                       # screen blend: light, not paint
    return f

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
