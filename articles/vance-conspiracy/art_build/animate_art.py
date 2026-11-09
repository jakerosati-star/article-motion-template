"""Vance Conspiracy art: a conspiracy board. Pinned cutouts sway, pulses of light run along the red strings, and
midway Vance leans in and shoves Trump off-centre before everything settles back. Paper behind moving pieces is
redrawn (grid period 101px). Seamless 10s loop; frame 0 is the original. Run layers.py first. Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image
from scipy import ndimage

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
PIECES = dict(np.load('pieces.npz'))
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

# clean paper plate under every piece that moves
def line_prof(v, ph): d = np.abs(((v - ph + 50.5) % 101) - 50.5); return np.exp(-(d ** 2) / (2 * 1.6 ** 2))
grid = np.maximum(line_prof(xx, 1270), line_prof(yy, 589))
noise = cv2.GaussianBlur(np.random.default_rng(1).normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.2) * 6
paper = np.array([249, 236, 232], np.float32) * (1 - 0.13 * grid)[..., None] + noise[..., None]
occ = np.max(np.stack(list(PIECES.values())), 0)
hole = ndimage.gaussian_filter(ndimage.binary_dilation(occ > 0.05, iterations=6).astype(np.float32), 2)
bg = a * (1 - hole[..., None]) + paper * hole[..., None]   # original art, with clean paper wherever a piece can move off

# red strings: thin, saturated red on the paper (not the hat)
R, G, B = a[..., 0], a[..., 1], a[..., 2]
red = ((R - G) > 90) & ((R - B) > 90) & (occ < 0.05)
strings = red & ~ndimage.binary_opening(red, iterations=6)
strings = ndimage.gaussian_filter(strings.astype(np.float32), 0.8)
diag1 = xx * 0.7 + yy * 0.7; diag2 = xx * 0.7 - yy * 0.7

LAYERS = {k: np.dstack([a, m * 255]).astype(np.float32) for k, m in PIECES.items()}
def cen(m): ys, xs = np.nonzero(m > 0.5); return float(xs.mean()), float(ys.mean())
CEN = {k: cen(m) for k, m in PIECES.items()}
ORDER = ['hat', 'pepe', 'rubio', 'thiel', 'tucker', 'donjr', 'trump', 'vance']   # back to front
PH = {k: i * 1.3 for i, k in enumerate(ORDER)}

def ss(x): x = max(0.0, min(1.0, x)); return x * x * (3 - 2 * x)
def wv(k, t, ph=0): return math.sin(TAU * k * t / P + ph)
def shove(t):
    """Two shoves. 1: Vance leans in from frame 1, Trump knocked aside by 1.0s, both settle back by ~3.6s.
    2: again at 4.6s, a bit harder; settle back 8.0-9.6s so the loop resets."""
    def one(t0, lean, hit, back0, back_len):
        v = ss((t - t0) / lean) * (1 - ss((t - back0) / back_len))
        tr = ss((t - t0 - lean * 0.8) / hit) * (1 - ss((t - back0 - 0.3) / back_len))
        dt = t - t0 - lean * 0.8 - hit
        wob = math.exp(-max(0, dt) * 3) * math.sin(max(0, dt) * 18) * (dt > 0)
        return v, tr, wob
    v1, t1, w1 = one(0.0, 0.6, 0.45, 2.4, 1.0)
    v2, t2, w2 = one(4.6, 0.55, 0.4, 8.0, 1.3)
    return max(v1, 1.15 * v2), max(t1, 1.15 * t2), w1 + w2

enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                       stdin=subprocess.PIPE)
for i in range(N):
    t = i / FPS
    if i == 0: enc.stdin.write(a.astype(np.uint8).tobytes()); Image.fromarray(a.astype(np.uint8)).save('_f000.png'); continue   # frame 0 = the original
    f = bg.copy()
    # light pulses along the strings (two diagonal sweeps per loop)
    if i:
        p1 = np.exp(-(((diag1 - (t / P) * 3400) % 1700 - 850) ** 2) / (2 * 60 ** 2))
        p2 = np.exp(-(((diag2 + 1100 - (t / P) * 3400) % 1700 - 850) ** 2) / (2 * 60 ** 2))
        g = strings * np.clip(p1 + p2, 0, 1)
        f = f + g[..., None] * np.array([90, 140, 120], np.float32)
    v, tr, wob = shove(t)
    for k in ORDER:
        cx, cy = CEN[k]; ang = 1.2 * wv(1, t, PH[k]) + 0.5 * wv(2, t, PH[k] * 2) - (1.2 * math.sin(PH[k]) + 0.5 * math.sin(PH[k] * 2))
        dx = dy = 0.0
        if k == 'vance': dx, dy, ang = dx - 70 * v, dy + 45 * v, ang + 7 * v
        if k == 'trump': dx, dy, ang = dx - 120 * tr, dy + 20 * tr, ang + 9 * tr + 3 * wob
        M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0); M[0, 2] += dx; M[1, 2] += dy
        L = cv2.warpAffine(LAYERS[k], M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        al = L[..., 3] / 255
        sh = cv2.GaussianBlur(al, (0, 0), 10); sh = np.roll(np.roll(sh, 8, 0), 6, 1) * 0.22 * (abs(dx) + abs(dy) > 1)   # soft drop shadow while lifted
        f = f * (1 - sh[..., None]) ; f = f * (1 - al[..., None]) + L[..., :3] * al[..., None]
    z = 1 + 0.03 * (0.5 - 0.5 * math.cos(TAU * t / P))
    Mc = np.float32([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H / 2]])
    out = cv2.warpAffine(np.clip(f, 0, 255).astype(np.uint8), Mc, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if i in (0, 30, 105, 165, 290, 299): Image.fromarray(out).save(f'_f{i:03d}.png')
    enc.stdin.write(out.tobytes())
enc.stdin.close(); enc.wait()
