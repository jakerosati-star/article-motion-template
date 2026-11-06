"""Trumpfall art: the figure slowly pushes forward against a counter-drifting collage (depth), film grain and
flicker throughout, and two short glitch bursts per loop (sliced, RGB-split frames) that 'break' the image.
Seamless 10s loop; frame 0 is the clean original. Writes ../art_loop.mp4"""
import math, random, subprocess
import numpy as np, cv2
from PIL import Image
from scipy import ndimage

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
m = np.array(Image.open('rembg_mask.png')).astype(np.float32) / 255
fig_a = np.where(ndimage.binary_erosion(ndimage.binary_fill_holes(m > 0.5), iterations=2), 1.0, m)
hole = cv2.dilate((fig_a > 0.05).astype(np.uint8), np.ones((31, 31), np.uint8))
bg = cv2.inpaint(a.astype(np.uint8), hole, 15, cv2.INPAINT_TELEA).astype(np.float32)
bg = np.where(cv2.dilate((fig_a > 0.05).astype(np.uint8), np.ones((5, 5), np.uint8))[..., None] > 0, bg, a)
FIG = np.dstack([a, fig_a * 255]).astype(np.float32)
ANCHOR = (1100.0, 1080.0)          # figure scales from the bottom, as if stepping toward camera

def wv(k, t, ph=0): return math.sin(TAU * k * t / P + ph)
def ease(t): return 0.5 - 0.5 * math.cos(TAU * t / P)          # 0 at the loop point, 1 mid-loop
def over(base, rgb, al): return base * (1 - al[..., None]) + rgb * al[..., None]

BURSTS = [(3.1, 0.30), (6.9, 0.40)]   # glitch bursts (start, length)
def glitch_amount(t):
    for t0, d in BURSTS:
        if t0 <= t < t0 + d: return 0.5 + 0.5 * math.sin(math.pi * (t - t0) / d)
    return 0.0

def glitch(img, amt, seed):
    r = random.Random(seed); out = img.copy()
    y = 0
    while y < H:                      # horizontal slices shoved sideways
        h = r.randint(8, 90)
        if r.random() < 0.55:
            out[y:y + h] = np.roll(img[y:y + h], int(r.uniform(-60, 60) * amt), axis=1)
        y += h
    sh = int(10 * amt)                # RGB split
    out[..., 0] = np.roll(out[..., 0], sh, axis=1); out[..., 2] = np.roll(out[..., 2], -sh, axis=1)
    if r.random() < 0.6:              # a hot band of the collage's magenta
        y0 = r.randint(0, H - 40); out[y0:y0 + r.randint(6, 26)] = np.array([230, 20, 200], np.float32)
    return out

enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                       stdin=subprocess.PIPE)
for i in range(N):
    t = i / FPS; e = ease(t)
    zb = 1 + 0.025 * e; dxb = 14 * wv(1, t)                       # collage drifts one way...
    Mb = np.array([[zb, 0, (1 - zb) * W / 2 + dxb], [0, zb, (1 - zb) * H / 2]], np.float32)
    f = cv2.warpAffine(bg, Mb, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    zf = 1 + 0.045 * e; dxf = -10 * wv(1, t)                      # ...the figure the other, and steps closer
    Mf = np.array([[zf, 0, (1 - zf) * ANCHOR[0] + dxf], [0, zf, (1 - zf) * ANCHOR[1]]], np.float32)
    g = cv2.warpAffine(FIG, Mf, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    f = over(f, g[..., :3], g[..., 3] / 255)
    if i > 0:                                                    # film grain + gentle exposure flicker (frame 0 stays clean)
        grain = np.random.default_rng(i).normal(0, 7, (H // 2, W // 2, 1)).astype(np.float32)
        f = f * (1 + 0.025 * wv(17, t) * wv(5, t, 1)) + cv2.resize(grain, (W, H))[..., None]
    amt = glitch_amount(t)
    if amt > 0: f = glitch(f, amt, i // 2)                       # glitch pattern changes every other frame
    out = np.clip(f, 0, 255).astype(np.uint8)
    if i in (0, 95, 100, 150, 212, 299): Image.fromarray(out).save(f'_f{i:03d}.png')
    enc.stdin.write(out.tobytes())
enc.stdin.close(); enc.wait()
