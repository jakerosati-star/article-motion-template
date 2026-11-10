"""Neo-Nazi Livestream art: the "Oh, nooo!" error dialog pulses like an alert in the first second (and again
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

enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                       stdin=subprocess.PIPE)
for i in range(N):
    t = i / FPS
    if i == 0: enc.stdin.write(a.astype(np.uint8).tobytes()); continue
    # tissue flutter (warp of the background only)
    fx = 7 * np.sin(yy * 0.012 + xx * 0.004 - TAU * 6 * t / P) * tissue
    fy = 4 * np.sin(xx * 0.01 - TAU * 4 * t / P + 1.0) * tissue
    f = cv2.remap(a, xx + fx, yy + fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # screen flicker + a soft rolling scan band
    band = np.exp(-(((yy - (t / P * 3 * 1300) % 1300 + 150)) ** 2) / (2 * 40 ** 2))
    fl = 1 + (0.05 * wv(23, t) * wv(7, t, 0.5) + 0.12 * band) * scr
    f = f * fl[..., None]
    # dialog alert pulse (only ever scales up, so nothing behind it shows)
    p = pulse(t); s = 1 + 0.07 * p
    if p > 0.001:
        M = np.float32([[s, 0, (1 - s) * DC[0]], [0, s, (1 - s) * DC[1]]])
        w = cv2.warpAffine(DLG, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0); al = w[..., 3] / 255
        f = f * (1 - al[..., None]) + w[..., :3] * al[..., None]
    else:
        f = f * (1 - dlg[..., None]) + a * dlg[..., None]
    f = f * (1 - man[..., None]) + a * man[..., None]        # the man stays put, in front
    # recording light: blinks twice a second with a red glow
    on = (math.sin(TAU * 2 * t - math.pi / 2) > 0)
    if on:
        g = np.zeros((H, W), np.float32); cv2.circle(g, LED, 9, 1.0, -1, cv2.LINE_AA)
        glow = cv2.GaussianBlur(g, (0, 0), 14) * 1.6 + g
        f = f + glow[..., None] * np.array([255, 30, 40], np.float32) * 0.9
    z = 1 + 0.03 * (0.5 - 0.5 * math.cos(TAU * t / P))
    Mc = np.float32([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H / 2]])
    out = cv2.warpAffine(np.clip(f, 0, 255).astype(np.uint8), Mc, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if i in (8, 15, 30, 100, 160, 299): Image.fromarray(out).save(f'_f{i:03d}.png')
    enc.stdin.write(out.tobytes())
enc.stdin.close(); enc.wait()
