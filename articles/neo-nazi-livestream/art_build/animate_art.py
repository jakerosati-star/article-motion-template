"""Neo-Nazi Livestream art: livestream-style reaction emoji (skulls, X's, hiding eyes...) float up the screen, the error dialog pulses like an alert in the first second (and again
mid-loop), the webcam's red recording light blinks with a glow, the screen flickers with a rolling scan band,
the tissue flutters, slow push-in. Seamless 10s loop; frame 0 is the original. Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw
from scipy import ndimage

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
PC = dict(np.load('pieces.npz')); man, dlg, scr = PC['man'], PC['dialog'], PC['screen']
MAN = np.dstack([a, man * 255]); DLG = np.dstack([a, dlg * 255])
DC = (1330.0, 815.0)                     # dialog centre
LED = (1423, 243)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
# tissue: flutter warp, strongest at the top of the tissue, zero on the man and away from the tissue
tm = Image.new('L', (W, H), 0); ImageDraw.Draw(tm).polygon([(100, 70), (840, 30), (1110, 200), (1060, 520), (150, 560)], fill=255)
tissue = ndimage.gaussian_filter(np.array(tm, np.float32) / 255, 30) * (1 - ndimage.gaussian_filter((man > 0.3).astype(np.float32), 12))
tissue *= np.clip((560 - yy) / 400, 0, 1)

def ss(x): x = max(0.0, min(1.0, x)); return x * x * (3 - 2 * x)
def wv(k, t, ph=0): return math.sin(TAU * k * t / P + ph)
PULSES = [(0.15, 0.32), (0.55, 0.32), (5.0, 0.32), (5.4, 0.32)]   # alert "boing"s
def pulse(t):
    s = 0.0
    for t0, d in PULSES:
        u = (t - t0) / d
        if 0 <= u <= 1: s = max(s, math.sin(math.pi * u) * math.exp(-u))
    return s

# --- "Oh, nooo!" keeps going: O's typed onto the text's slant, then backspaced before the loop point ---
lum = a.mean(2)
TX0, TX1, TY0, TY1 = 1359, 1519, 826, 938          # the original "ooo!" area
src = a.copy(); src[TY0:TY1, TX0:TX1] = a[TY0:TY1, TX0 + 178:TX1 + 178]   # screen grid repeats every ~22.2px; 178 = 8 cells
txt = np.zeros((H, W), np.float32); txt[TY0:TY1, TX0:TX1] = (lum[TY0:TY1, TX0:TX1] < 150)
txt = cv2.dilate(txt, np.ones((7, 7), np.uint8))
def nblur(img, w, sig):   # blur that ignores masked-out (text) pixels
    num = cv2.GaussianBlur(img * w[..., None], (0, 0), sig); den = cv2.GaussianBlur(w, (0, 0), sig)[..., None]
    return num / np.maximum(den, 1e-4)
keep = 1 - txt
tone = nblur(a, keep, 12) - nblur(src, np.ones_like(keep), 12)       # match local brightness / colour
fill = src + tone
fm = cv2.GaussianBlur(txt, (0, 0), 1.5)[..., None]
clean = a * (1 - fm) + fill * fm
def glyph(x0, y0, x1, y1):
    m = np.clip((140 - lum[y0:y1, x0:x1]) / 60, 0, 1); return m, (x0, y0)
G_O = glyph(1447, 852, 1497, 930)                   # third "o"
G_BANG = glyph(1494, 832, 1515, 924)                # "!"
O_STEP = (45.0, -6.25)                              # each o: 45px right, 6.25px up (the text's slant)
O_XY = [(1361 - 1451, 874 - 861), (1406 - 1451, 868 - 861), (0, 0)]   # o1..o3 relative to the o3 glyph
TYPE_OS = False   # the "Oh, noooo..." typing gag was tried and dropped; the text stays as drawn
def n_os(t):
    if not TYPE_OS: return 3
    n = 3 + int(max(0, min(14, (t - 0.35) / 0.11 + 1)))           # one more O every 0.11s, up to 17
    if t > 8.6: n = max(3, n - int((t - 8.6) / 0.05) - 1)          # quick backspace
    return n
def draw_text(img, n):
    if n == 3: return a.copy()
    out = clean.copy()
    def stamp(g, dx, dy):
        m, (gx, gy) = g; x, y = gx + int(round(dx)), gy + int(round(dy)); h, w = m.shape
        xa, xb = max(0, x), min(W, x + w)
        if xb <= xa: return
        mm = m[:, xa - x:xb - x]
        out[y:y + h, xa:xb] = out[y:y + h, xa:xb] * (1 - mm[..., None]) + np.array([8, 8, 12], np.float32) * mm[..., None]
    for k in range(n):
        dx, dy = O_XY[k] if k < 3 else ((k - 2) * O_STEP[0], (k - 2) * O_STEP[1])
        stamp(G_O, dx, dy)
    stamp(G_BANG, (n - 3) * O_STEP[0], (n - 3) * O_STEP[1])
    return out

# --- livestream-style reactions floating up the screen ---
from PIL import ImageFont
EF = ImageFont.truetype('/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf', 109)   # Noto Color Emoji (OFL)
def emoji_img(ch, size):
    im = Image.new('RGBA', (140, 140)); ImageDraw.Draw(im).text((4, 4), ch, font=EF, embedded_color=True)
    im = im.crop(im.getbbox()).resize((size, size), Image.LANCZOS); return np.array(im).astype(np.float32)
EMO = {c: emoji_img(c, 96) for c in ['💀', '😱', '🙈', '❌', '🤢', '😬', '🫣', '🤮', '😳']}
import random as _r
_rr = _r.Random(11); REACT = []
seq = ['😱', '💀', '🙈', '❌', '🤮', '🫣', '💀', '😬', '🤢', '❌', '😳', '🙈', '💀', '🫣']
for j, ch in enumerate(seq):
    REACT.append((ch, 0.5 + j * 0.5 + _rr.uniform(-0.12, 0.12), _rr.uniform(1480, 1780), _rr.uniform(-40, 40), _rr.uniform(0.85, 1.15)))
def draw_reactions(f, t):
    for ch, t0, x0, drift, sc in REACT:
        u = (t - t0) / 2.4
        if not 0 < u < 1: continue
        y = 1010 - 560 * (1 - (1 - u) ** 2); x = x0 + drift * u + 16 * math.sin(u * 7 + x0)
        pop = min(1.0, u / 0.08); size = sc * (0.5 + 0.5 * pop + 0.15 * math.sin(math.pi * min(1, u / 0.12)))
        alpha = min(1.0, (1 - u) / 0.25)
        e = EMO[ch]; sz = max(8, int(96 * size)); e = cv2.resize(e, (sz, sz), interpolation=cv2.INTER_LINEAR)
        xi, yi = int(x - sz / 2), int(y - sz / 2); xa, xb, ya, yb = max(0, xi), min(W, xi + sz), max(0, yi), min(H, yi + sz)
        if xb <= xa or yb <= ya: continue
        ee = e[ya - yi:yb - yi, xa - xi:xb - xi]; al = ee[..., 3:4] / 255 * alpha
        f[ya:yb, xa:xb] = f[ya:yb, xa:xb] * (1 - al) + ee[..., :3] * al

enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                       stdin=subprocess.PIPE)
for i in range(N):
    t = i / FPS
    if i == 0: enc.stdin.write(a.astype(np.uint8).tobytes()); continue
    # tissue flutter (warp of the background only)
    base = draw_text(a, n_os(t)); DLG_t = np.dstack([base, dlg * 255])
    fx = 7 * np.sin(yy * 0.012 + xx * 0.004 - TAU * 6 * t / P) * tissue
    fy = 4 * np.sin(xx * 0.01 - TAU * 4 * t / P + 1.0) * tissue
    f = cv2.remap(base, xx + fx, yy + fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # screen flicker + a soft rolling scan band
    band = np.exp(-(((yy - (t / P * 3 * 1300) % 1300 + 150)) ** 2) / (2 * 40 ** 2))
    fl = 1 + (0.05 * wv(23, t) * wv(7, t, 0.5) + 0.12 * band) * scr
    f = f * fl[..., None]
    # dialog alert pulse (only ever scales up, so nothing behind it shows)
    p = pulse(t); s = 1 + 0.07 * p
    if p > 0.001:
        M = np.float32([[s, 0, (1 - s) * DC[0]], [0, s, (1 - s) * DC[1]]])
        w = cv2.warpAffine(DLG_t, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0); al = w[..., 3] / 255
        f = f * (1 - al[..., None]) + w[..., :3] * al[..., None]
    else:
        f = f * (1 - dlg[..., None]) + base * dlg[..., None]
    f = f * (1 - man[..., None]) + a * man[..., None]        # the man stays put, in front
    # recording light: blinks twice a second with a red glow
    on = (math.sin(TAU * 2 * t - math.pi / 2) > 0)
    if on:
        g = np.zeros((H, W), np.float32); cv2.circle(g, LED, 9, 1.0, -1, cv2.LINE_AA)
        glow = cv2.GaussianBlur(g, (0, 0), 14) * 1.6 + g
        f = f + glow[..., None] * np.array([255, 30, 40], np.float32) * 0.9
    draw_reactions(f, t)
    z = 1 + 0.03 * (0.5 - 0.5 * math.cos(TAU * t / P))
    Mc = np.float32([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H / 2]])
    out = cv2.warpAffine(np.clip(f, 0, 255).astype(np.uint8), Mc, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if i in (20, 45, 90, 180, 268, 299): Image.fromarray(out).save(f'_f{i:03d}.png')
    enc.stdin.write(out.tobytes())
enc.stdin.close(); enc.wait()
