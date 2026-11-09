"""Democratic Party Project art: the six markers take turns. On each turn the pen scribbles and, on the four dark
circles, a fresh pass of ink glides around the circle from the pen tip. Idle pens bob gently. Seamless 10s loop,
frame 0 is the original art. Run layers.py first. Writes ../art_loop.mp4"""
import math, json, subprocess
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
pens = dict(np.load('pens.npz')); ink = np.load('ink.npy'); ell = json.load(open('ellipses.json'))
TIPS = {'A': ((499, 435), (-60, -40)), 'B': ((838, 154), (965, -30)), 'C': ((1295, 109), (1400, -30)),
        'D': ((1414, 474), (1960, 50)), 'E': ((1626, 845), (1960, 440)), 'F': ((909, 896), (745, 1100))}
TURNS = [('B', 0.6), ('A', 2.8), ('D', 5.0), ('E', 7.2)]; TURN_LEN = 1.6
MOVING = ('A', 'B', 'D', 'E')   # the light-blue pens (C, F) stay put: their tracing is too rough to move cleanly

pen_all = np.max(np.stack([pens[k] for k in MOVING]), 0)
hole = cv2.dilate((pen_all > 0.05).astype(np.uint8), np.ones((19, 19), np.uint8))
bg = cv2.inpaint(a.astype(np.uint8), hole, 9, cv2.INPAINT_TELEA).astype(np.float32)
LAYERS = {k: np.dstack([a, pens[k] * 255]).astype(np.float32) for k in MOVING}

# ink path per dark circle: ellipse outline, starting at the point nearest its pen's tip
S = 4; hs, ws = H // S, W // S; inks = cv2.resize(ink, (ws, hs), interpolation=cv2.INTER_AREA)
ys, xs = np.mgrid[0:hs, 0:ws].astype(np.float32)
PATH = {}
for k, (ex, ey, r1, r2, ang) in ell.items():
    pts = np.array(cv2.ellipse2Poly((int(ex), int(ey)), (int(r1), int(r2)), int(ang), 0, 360, 2), np.float32)
    tip = np.array(TIPS[k][0], np.float32); i0 = int(np.argmin(((pts - tip) ** 2).sum(1)))
    PATH[k] = np.roll(pts, -i0, axis=0)

def clamp(x): return max(0.0, min(1.0, x))
def ss(x): x = clamp(x); return x * x * (3 - 2 * x)
def wv(k, t, ph=0): return math.sin(TAU * k * t / P + ph)
def over(base, rgb, al): return base * (1 - al[..., None]) + rgb * al[..., None]

enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                       stdin=subprocess.PIPE)
for i in range(N):
    t = i / FPS
    f = bg.copy()
    # fresh ink pass on the active dark circle
    for k, t0 in TURNS:
        u = (t - t0) / TURN_LEN
        if k in PATH and 0 < u < 1.25:
            path = PATH[k]; n = len(path); head = ss(min(u, 1.0)) * n
            g = np.zeros((hs, ws), np.float32)
            for j in range(16):                          # glowing head + fading trail
                p = path[int(head - j * n / 40) % n] / S; w = (1 - j / 16) * (1 - clamp((u - 1.0) / 0.25))
                if w > 0: g += w * np.exp(-((xs - p[0]) ** 2 + (ys - p[1]) ** 2) / (2 * 9.0 ** 2))
            g = np.clip(g, 0, 1) * inks
            g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
            f = over(f, np.array([40, 95, 255], np.float32), 0.6 * g)                  # fresh, saturated ink
            f = np.clip(f + 70 * g[..., None] ** 3, 0, 255)                           # wet sheen at the head
    # pens: idle bob along their axis; scribble during their turn
    for k in MOVING:
        (tx, ty), (ex2, ey2) = TIPS[k]
        d = np.array([ex2 - tx, ey2 - ty], np.float32); uax = d / np.linalg.norm(d); nrm = np.array([-uax[1], uax[0]])
        ph = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}[k]
        off = uax * 2.0 * wv(2, t, ph)
        for kk, t0 in TURNS:
            if kk == k:
                v = (t - t0) / TURN_LEN
                if 0 < v < 1:
                    env = math.sin(math.pi * v); ang = TAU * 2.5 * v
                    off = off + env * (uax * 5 * math.cos(ang) + nrm * 8 * math.sin(ang)) - uax * 6 * env   # leans in and scribbles
        M = np.float32([[1, 0, off[0]], [0, 1, off[1]]])
        Lw = cv2.warpAffine(LAYERS[k], M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        f = over(f, Lw[..., :3], Lw[..., 3] / 255)
    z = 1 + 0.03 * (0.5 - 0.5 * math.cos(TAU * t / P))
    Mc = np.float32([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H / 2]])
    out = cv2.warpAffine(np.clip(f, 0, 255).astype(np.uint8), Mc, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if i in (0, 40, 110, 190, 255, 299): Image.fromarray(out).save(f'_f{i:03d}.png')
    enc.stdin.write(out.tobytes())
enc.stdin.close(); enc.wait()
