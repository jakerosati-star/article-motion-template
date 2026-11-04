"""Animate the AOC collage into a seamless 10s loop (../art_loop.mp4). Run layers.py first."""
import math, random, subprocess
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
bg = np.array(Image.open('L_bg.png').convert('RGB')).astype(np.float32)
mp = np.array(Image.open('L_map.png')).astype(np.float32)
fg = np.array(Image.open('L_fg.png')).astype(np.float32)
b = np.load('band.npy'); BX0, BY0, BX1, BY1 = [int(v) for v in b[:4]]; BAND = b[4:].astype(np.float32)
CLIP_Y = 962                       # AOC's bottom edge: the group rises from / sinks behind this line
MAP_C = (1450, 540)                # map scale centre
FG_ANCHOR = (1000, CLIP_Y)         # breathing scale anchored on the bottom edge

def clamp(x): return max(0.0, min(1.0, x))
def eoq(t): return 1 - (1 - t) ** 5
def eoc(t): return 1 - (1 - t) ** 3
def eic(t): return t ** 3
def ease_io(t): return t * t * (3 - 2 * t)
def wv(k, t, ph=0): return math.sin(TAU * k * t / P + ph)

def warp(rgba, scale, cx, cy, dx=0, dy=0):
    M = np.array([[scale, 0, (1 - scale) * cx + dx], [0, scale, (1 - scale) * cy + dy]], np.float32)
    return cv2.warpAffine(rgba, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def over(base, rgb, a): return base * (1 - a[..., None]) + rgb * a[..., None]

# twinkling speckles (periodic)
rng = random.Random(3)
dust = [dict(x=rng.uniform(0, W), y=rng.uniform(0, H), r=rng.choice([1, 1, 1.5, 2, 2.5]), k=rng.randint(1, 4),
             ph=rng.uniform(0, TAU), orb=rng.uniform(3, 9), m=rng.choice([1, 2])) for _ in range(170)]

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
diag = (xx * 0.8 + yy * 0.6)       # sheen direction

enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                       stdin=subprocess.PIPE)
for i in range(N):
    t = i / FPS
    f = bg.copy()
    # speckles
    dl = np.zeros((H, W), np.float32)
    for d in dust:
        a = 0.5 + 0.5 * wv(d['k'], t, d['ph'])
        x = d['x'] + d['orb'] * math.cos(TAU * d['m'] * t / P + d['ph']); y = d['y'] + d['orb'] * math.sin(TAU * d['m'] * t / P + d['ph'])
        cv2.circle(dl, (int(x * 4), int(y * 4)), int(d['r'] * 4), a * 0.8, -1, lineType=cv2.LINE_AA, shift=2)
    f = over(f, np.array([255, 255, 252], np.float32), dl)

    # band: opens from the centre, closes at the end
    p = 1.0                                     # band is on screen the whole loop
    if p > 0:
        cx = (BX0 + BX1) / 2; hw = (BX1 - BX0) / 2 * p
        ba = np.zeros((H, W), np.float32); ba[BY0:BY1, int(cx - hw):int(cx + hw)] = 1
        f = over(f, BAND, ba)

    # map: grows in behind AOC, gentle drift, sheen pass, shrinks out
    q = eoc(clamp((t - 0.05) / 0.8)); qo = eic(clamp((t - 9.2) / 0.5))   # grows in right away, out just before the loop point
    if q > 0 and qo < 1:
        sc = (0.86 + 0.14 * q) * (1 - 0.12 * qo)
        m = warp(mp, sc, *MAP_C, dx=6 * wv(1, t, 0.3) - 6 * math.sin(0.3))
        al = m[..., 3] / 255 * min(1, q * 1.6) * (1 - qo)
        rgb = m[..., :3]
        sp = (t - 4.2) / 1.6                       # sheen sweeps across between 4.2s and 5.8s
        if 0 < sp < 1:
            pos = -200 + sp * (W * 0.8 + H * 0.6 + 400)
            rgb = np.clip(rgb + 70 * np.exp(-((diag - pos) ** 2) / (2 * 90 ** 2))[..., None], 0, 255)
        f = over(f, rgb, al)

    # AOC + voters: rise from behind the clip line, breathe, sink out
    # on screen from frame 0: starts slightly large and settles in, eases back up before the loop point
    if True:
        rise = 0
        settle = 0.03 * (1 - eoc(clamp(t / 1.2))) + 0.03 * ease_io(clamp((t - 8.9) / 1.1))
        breathe = 1 + settle + 0.014 * (0.5 - 0.5 * math.cos(TAU * t / P))
        g = warp(fg, breathe, *FG_ANCHOR, dy=rise)
        al = g[..., 3] / 255; al[CLIP_Y:] = 0
        f = over(f, g[..., :3], al)

    # camera: slow push-in and back
    z = 1 + 0.035 * (0.5 - 0.5 * math.cos(TAU * t / P))
    M = np.array([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H * 0.45]], np.float32)
    out = cv2.warpAffine(np.clip(f, 0, 255).astype(np.uint8), M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if i % 15 == 0 or i == N - 1: Image.fromarray(out).save(f'_f{i:03d}.png')
    enc.stdin.write(out.tobytes())
enc.stdin.close(); enc.wait()
