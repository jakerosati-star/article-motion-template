"""Dark Lesson art: flames lick upward and flicker behind the survivorship-bias plane, embers rise, the bullet holes
plot in one by one in the first seconds (each with a small flash), slow push-in; the "I Voted" border holds still.
Seamless 10s loop; frame 0 is the plane and flames with no holes yet (the holes are what gets plotted). Run layers.py first. Writes ../art_loop.mp4"""
import math, random, subprocess
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
fire = np.array(Image.open('fire.png').convert('RGB')).astype(np.float32)
PC = dict(np.load('pieces.npz')); lines, dots_m, inner = PC['lines'], PC['dots'], PC['inner']
DOTS = np.load('dots.npy')
X0, Y0, X1, Y1 = 108, 91, 1811, 989
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
hot = np.clip((fire.max(2) - fire.min(2) - 90) / 120, 0, 1) * inner
hot = cv2.GaussianBlur(hot, (0, 0), 4)
edge = np.clip(np.minimum.reduce([xx - X0, X1 - xx, yy - Y0, Y1 - yy]) / 40, 0, 1) * inner   # flames calm at the frame edge

rng = np.random.default_rng(4)
WAVES = [(rng.uniform(0.010, 0.022), rng.uniform(0.004, 0.012), rng.uniform(0, TAU), int(rng.integers(2, 5))) for _ in range(5)]
def flow(t):
    dx = np.zeros((H, W), np.float32); dy = np.zeros((H, W), np.float32)
    for ky, kx, ph, nn in WAVES:
        arg = yy * ky + xx * kx + ph + TAU * nn * t / P
        dx += 8 * np.sin(arg); dy += 9 * np.cos(arg * 0.8 + 1.1)
    return dx / 2.5 * edge, dy / 2.5 * edge
dx0, dy0 = flow(0.0)

# bullet holes: hidden at the start, plotted one by one (random order) from 0.1s to 3.6s
order = np.random.default_rng(7).permutation(len(DOTS))
T_DOT = {int(k): 0.1 + 3.5 * j / len(DOTS) for j, k in enumerate(order)}
dot_lab = np.zeros((H, W), np.int32)
for k, (cx, cy, r) in enumerate(DOTS):
    cv2.circle(dot_lab, (int(cx), int(cy)), int(r + 3), k + 1, -1)

er = random.Random(5); Ly = 900
embers = [dict(x=er.uniform(X0 + 40, X1 - 40), y0=er.uniform(0, Ly), n=er.choice([1, 2, 2, 3]), sw=er.uniform(8, 24), m=er.randint(1, 3),
               ph=er.uniform(0, TAU), r=er.choice([2, 2, 2.5, 3, 3.5]), k=er.randint(3, 8)) for _ in range(60)]
def wv(k, t, ph=0): return math.sin(TAU * k * t / P + ph)

enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                       stdin=subprocess.PIPE)
for i in range(N):
    t = i / FPS
    dx, dy = flow(t); dx -= dx0; dy -= dy0
    f = cv2.remap(fire, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    f = f * (1 + (0.07 * wv(11, t) * wv(4, t, 1.3) + 0.04 * wv(17, t, 0.4)) * hot)[..., None]
    el = np.zeros((H, W), np.float32)
    for e in embers:
        y = Y1 - 20 - ((e['y0'] + Ly * e['n'] * t / P) % Ly)
        x = e['x'] + e['sw'] * wv(e['m'], t, e['ph'])
        al = (0.55 + 0.45 * wv(e['k'], t, e['ph'])) * min(1, (y - Y0) / 120)
        if al > 0: cv2.circle(el, (int(x * 4), int(y * 4)), int(e['r'] * 4), al, -1, cv2.LINE_AA, 2)
    f = f + (cv2.GaussianBlur(el, (0, 0), 3) * 0.8 + el)[..., None] * np.array([255, 190, 70], np.float32)
    f = f * inner[..., None] + a * (1 - inner[..., None])                     # border stays exactly as drawn
    f = f * (1 - lines[..., None]) + a * lines[..., None]                     # plane outline
    # dots: visible once plotted; a white flash ring as each lands
    vis = np.zeros(len(DOTS) + 1, np.float32); flash = np.zeros(len(DOTS) + 1, np.float32)
    for k, t0 in T_DOT.items():
        u = (t - t0) / 0.35
        vis[k + 1] = (1.0 if u >= 0 else 0.0) * (1 - min(1.0, max(0.0, (t - 9.2) / 0.6)))   # all fade out before the loop point
        flash[k + 1] = max(0.0, 1 - u) if 0 <= u < 1 else 0.0
    dv = vis[dot_lab] * dots_m
    f = f * (1 - dv[..., None]) + a * dv[..., None]
    fl = cv2.GaussianBlur(flash[dot_lab] * (dot_lab > 0), (0, 0), 6) * 2.2
    f = f + fl[..., None] * np.array([255, 245, 220], np.float32) * 0.8
    z = 1 + 0.035 * (0.5 - 0.5 * math.cos(TAU * t / P))
    Mc = np.float32([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H * 0.45]])
    out = cv2.warpAffine(np.clip(f, 0, 255).astype(np.uint8), Mc, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if i in (3, 30, 60, 120, 299): Image.fromarray(out).save(f'_f{i:03d}.png')
    enc.stdin.write(out.tobytes())
enc.stdin.close(); enc.wait()
