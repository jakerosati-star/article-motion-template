"""Schumer vs. Van Hollen art: an awkward glance between the two. The camera pushes in on Schumer (a glint sweeps
across his glasses on the way in and again up close), then glides across to Van Hollen, lingers, and pulls back
out to the full frame for the loop point (frame 0 = the art). The clouds behind Van Hollen drift throughout.
Works from the full-size 3000px still so close-ups stay sharp. Run from art_build/ (after lenses.py).
Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
full = np.array(Image.open('../art_still.jpg').convert('RGB')).astype(np.float32)
H3, W3 = full.shape[:2]; SC = W3 / W                    # full-size px per 1920 px
art = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
up = lambda m: cv2.resize(m.astype(np.float32), (W3, H3), interpolation=cv2.INTER_LINEAR)

# --- sky behind Van Hollen (1920 coords): the blue panel minus his (pure grayscale) photo
PX0, PY0, PX1, PY1 = 1308, 150, 1814, 880
gray = (np.abs(art[..., 0] - art[..., 1]) <= 6) & (np.abs(art[..., 1] - art[..., 2]) <= 6)
sky = np.zeros((H, W), np.uint8); sky[PY0:PY1, PX0:PX1] = 1; sky[gray] = 0
sky = cv2.morphologyEx(sky, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(1 - sky[PY0:PY1, PX0:PX1], 4)
for i in range(1, n):                                   # small enclosed white clouds belong to the sky
    if st[i, 4] < 3000: sky[PY0:PY1, PX0:PX1][lab == i] = 1
sky3 = (up(sky) > 0.5).astype(np.float32)
sky3 = cv2.GaussianBlur(sky3, (0, 0), 1.0) * sky3       # keep his edge crisp
Q0, Q1, Q2, Q3 = [int(round(v * SC)) for v in (PX0, PY0, PX1, PY1)]
crop = full[Q1:Q3, Q0:Q2].astype(np.uint8)
hole = cv2.dilate(((1 - sky3[Q1:Q3, Q0:Q2]) > 0.5).astype(np.uint8), np.ones((13, 13), np.uint8)) * 255
sky_src = cv2.inpaint(crop, hole, 9, cv2.INPAINT_TELEA).astype(np.float32)    # clouds continue behind him
cy_, cx_ = np.mgrid[0:Q3 - Q1, 0:Q2 - Q0].astype(np.float32)
DRIFT = 70.0                                            # full-size px

# --- glasses glint (computed at 1920, it's soft light; scaled up)
lens = cv2.GaussianBlur(np.load('lenses.npy').astype(np.float32), (0, 0), 1.0)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
GLINTS = [0.45, 2.7]; SWEEP = 0.5
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

# --- camera keyframes (1920 coords): time, centre x, centre y, zoom
KEYS = [(0.0, 960, 540, 1.0),
        (2.0, 1095, 440, 1.68),     # push in on Schumer
        (3.4, 1108, 446, 1.72),     # linger
        (4.8, 1420, 480, 1.75),     # glide across to Van Hollen (held inside the frame edge)
        (7.0, 1430, 486, 1.78),     # linger
        (9.4, 960, 540, 1.0),       # pull back out
        (10.0, 960, 540, 1.0)]
def camera(t):
    for (t0, x0, y0, z0), (t1, x1, y1, z1) in zip(KEYS, KEYS[1:]):
        if t0 <= t <= t1:
            u = (t - t0) / (t1 - t0); u = u * u * (3 - 2 * u)            # ease in and out
            z = z0 + (z1 - z0) * u
            x = x0 + (x1 - x0) * u; y = y0 + (y1 - y0) * u
            x = min(max(x, W / 2 / z), W - W / 2 / z); y = min(max(y, H / 2 / z), H - H / 2 / z)   # stay inside the art
            return x, y, z
    return W / 2, H / 2, 1.0

def frame(t):
    f = full.copy()
    dx = DRIFT * (0.5 - 0.5 * math.cos(TAU * t / P))
    moved = cv2.remap(sky_src, cx_ - dx, cy_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    a = sky3[Q1:Q3, Q0:Q2, None]
    f[Q1:Q3, Q0:Q2] = f[Q1:Q3, Q0:Q2] * (1 - a) + moved * a
    g = glint(t)
    if g.max() > 0: f = 255 - (255 - f) * (1 - up(g)[..., None] * 0.95)    # screen blend: light, not paint
    x, y, z = camera(t)
    k = z / SC
    M = np.float32([[k, 0, W / 2 - k * x * SC], [0, k, H / 2 - k * y * SC]])
    return cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

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
