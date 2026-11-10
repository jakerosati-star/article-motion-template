"""Defund WAR art: a red DEFUNDED stamp slams onto the money map in the first second and fades before the loop point;
the strait's water shimmers, the money collage drifts, Trump and Hegseth push in from opposite edges.
Seamless 10s loop; frame 0 is the original. Run layers.py first. Writes ../art_loop.mp4"""
import math, subprocess, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
PC = dict(np.load('pieces.npz'))
TR = np.dstack([a, PC['trump'] * 255]).astype(np.float32); HG = np.dstack([a, PC['hegseth'] * 255]).astype(np.float32)
water = PC['water']
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

# --- the stamp: condensed caps in a double-ruled box, rotated, with ink-starved texture ---
FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'fonts', 'SpecialGothic-VariableFont_wdthwght.ttf')
f = ImageFont.truetype(FONT, 172); f.set_variation_by_axes([700, 75])
txt = 'DEFUNDED'; bb = f.getbbox(txt); tw, th = bb[2] - bb[0], bb[3] - bb[1]
pad = 46; SW, SH = tw + 2 * pad, th + 2 * pad
st = Image.new('L', (SW + 40, SH + 40), 0); d = ImageDraw.Draw(st)
d.rounded_rectangle((20, 20, 20 + SW, 20 + SH), radius=18, outline=255, width=16)
d.rounded_rectangle((44, 44, SW - 4, SH - 4), radius=10, outline=255, width=5)
d.text((20 + pad - bb[0], 20 + pad - bb[1]), txt, font=f, fill=255)
st = st.rotate(-8, expand=True, resample=Image.BICUBIC)
sm = np.array(st, np.float32) / 255
rng = np.random.default_rng(3)
grain = cv2.GaussianBlur(rng.random(sm.shape).astype(np.float32), (0, 0), 1.4)
blot = cv2.GaussianBlur(rng.random(sm.shape).astype(np.float32), (0, 0), 14)
sm *= np.clip((grain - 0.08) * 4, 0, 1) * np.clip((blot - 0.3) * 4, 0.7, 1)   # slightly worn, uneven ink
STAMP_C = (915, 560); RED = np.array([214, 18, 36], np.float32)

def ss(x): x = max(0.0, min(1.0, x)); return x * x * (3 - 2 * x)
def wv(k, t, ph=0): return math.sin(TAU * k * t / P + ph)
def stamp_state(t):
    """scale and opacity: drops in from 1.6x (0.15-0.45s), small bounce, holds, fades 7.6-8.8s"""
    if t < 0.15: return None
    u = (t - 0.15) / 0.3
    if u < 1: sc = 1.6 - 0.6 * (u * u); op = min(1, u * 1.5)
    else:
        v = t - 0.45; sc = 1 - 0.04 * math.exp(-v * 9) * math.cos(v * 40); op = 1
    op *= 1 - ss((t - 7.6) / 1.2)
    return (sc, 0.96 * op) if op > 0 else None

def put(f, layer, M):
    w = cv2.warpAffine(layer, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0); al = w[..., 3] / 255
    return f * (1 - al[..., None]) + w[..., :3] * al[..., None]

enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                       stdin=subprocess.PIPE)
for i in range(N):
    t = i / FPS
    if i == 0: enc.stdin.write(a.astype(np.uint8).tobytes()); continue
    e = 0.5 - 0.5 * math.cos(TAU * t / P)
    # background: slow drift + zoom of the money map, water ripples
    zb = 1 + 0.03 * e; dxb = 14 * wv(1, t) 
    rip = (2.2 * np.sin(xx * 0.035 + yy * 0.02 + TAU * 4 * t / P) + 1.6 * np.sin(yy * 0.05 - TAU * 3 * t / P)) * water
    mx = (xx - 960 - dxb) / zb + 960 + rip; my = (yy - 540) / zb + 540 + 0.6 * rip
    fr = cv2.remap(a, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fr = fr + (14 * np.clip(np.sin(xx * 0.06 + yy * 0.03 + TAU * 5 * t / P), 0, 1) ** 8 * water)[..., None]   # glints on the water
    # stamp (under the faces)
    s = stamp_state(t)
    if s:
        sc, op = s; h, w = sm.shape
        M = np.float32([[sc, 0, STAMP_C[0] - sc * w / 2], [0, sc, STAMP_C[1] - sc * h / 2]])
        al = cv2.warpAffine(sm, M, (W, H), flags=cv2.INTER_LINEAR) * op
        fr = fr * (1 - al[..., None]) + RED * al[..., None]
    # faces push in from their outer corners (only ever larger, so nothing behind them shows)
    sT = 1 + 0.05 * e; sH = 1 + 0.05 * (0.5 - 0.5 * math.cos(TAU * t / P - 0.6) - (0.5 - 0.5 * math.cos(-0.6)) * (1 - e))
    fr = put(fr, TR, np.float32([[sT, 0, (1 - sT) * 0], [0, sT, (1 - sT) * 1080]]))
    fr = put(fr, HG, np.float32([[max(1, sH), 0, (1 - max(1, sH)) * 1920], [0, max(1, sH), (1 - max(1, sH)) * 1080]]))
    out = np.clip(fr, 0, 255).astype(np.uint8)
    if i in (6, 12, 30, 150, 270, 299): Image.fromarray(out).save(f'_f{i:03d}.png')
    enc.stdin.write(out.tobytes())
enc.stdin.close(); enc.wait()
