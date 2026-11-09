"""Bar Crawl art: the shaker gets a few sharp shakes, the drinks ripple and catch a glint, the election screen
flickers and LIVE pulses, slow push-in. Warps only (no cutouts), so nothing leaves a hole.
Seamless 10s loop; frame 0 is the original. Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw
from scipy import ndimage

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
def poly(pts, blur):
    m = Image.new('L', (W, H), 0); ImageDraw.Draw(m).polygon(pts, fill=255)
    return ndimage.gaussian_filter(np.array(m, np.float32) / 255, blur)
def ell(cx, cy, rx, ry, blur):
    m = Image.new('L', (W, H), 0); ImageDraw.Draw(m).ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
    return ndimage.gaussian_filter(np.array(m, np.float32) / 255, blur)

shaker = poly([(470, 380), (760, 380), (900, 520), (930, 890), (700, 910), (620, 820), (450, 650)], 45)   # hands + shaker, soft falloff up the arms
glasses = [poly([(795, 512), (1068, 512), (935, 700)], 4), ell(1378, 545, 128, 85, 4), poly([(1478, 508), (1752, 508), (1615, 690)], 4)]
glass_all = np.clip(sum(glasses), 0, 1)
screen = poly([(760, 0), (1850, 0), (1850, 815), (900, 815), (900, 520), (760, 520)], 8) * (1 - glass_all)
live = ell(1680, 573, 75, 26, 3)
diag = xx * 0.8 + yy * 0.6

def clamp(x): return max(0.0, min(1.0, x))
def wv(k, t, ph=0): return math.sin(TAU * k * t / P + ph)
SHAKES = [(1.4, 0.9), (5.6, 1.1)]   # (start, length) of shake bursts
def shake(t):
    for t0, d in SHAKES:
        u = (t - t0) / d
        if 0 <= u <= 1:
            env = math.sin(math.pi * u) ** 0.6
            s = env * math.sin(TAU * 7 * u)                     # ~7 shakes per burst
            return 9 * s, -11 * s                               # along the shaker's diagonal
    return 0.0, 0.0
GLINTS = [(2.6, 0), (3.0, 1), (3.4, 2), (7.4, 0), (7.8, 1), (8.2, 2)]

enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                       stdin=subprocess.PIPE)
for i in range(N):
    t = i / FPS
    sx, sy = shake(t)
    rip = 1.6 * np.sin(xx * 0.05 + yy * 0.02 + TAU * 3 * t / P) * glass_all * (1 if i else 0)     # drinks ripple
    mx = xx - sx * shaker; my = yy - sy * shaker + rip
    f = cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # screen: gentle broadcast flicker + LIVE pulse (exactly 1.0 at t = 0)
    fl = 1 + 0.035 * (wv(13, t) * wv(5, t, 1.2) + 0.5 * wv(29, t, 0.3) - 0.5 * math.sin(0.3))
    pulse = 0.5 - 0.5 * math.cos(TAU * 5 * t / P)
    f = f * (1 + (fl - 1) * screen)[..., None] + (70 * pulse * live)[..., None] * np.array([1, 0.85, 1], np.float32)
    # glints across the glasses, one after another
    for t0, g in GLINTS:
        u = (t - t0) / 0.5
        if 0 < u < 1:
            x0 = [790, 1245, 1475][g]; pos = (x0 - 80) * 0.8 + 450 + u * 380
            f = f + (90 * np.exp(-((diag - pos) ** 2) / (2 * 22 ** 2)) * glasses[g])[..., None]
    z = 1 + 0.035 * (0.5 - 0.5 * math.cos(TAU * t / P))
    Mc = np.float32([[z, 0, (1 - z) * W * 0.55], [0, z, (1 - z) * H * 0.5]])
    out = cv2.warpAffine(np.clip(f, 0, 255).astype(np.uint8), Mc, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if i in (0, 48, 51, 90, 232, 299): Image.fromarray(out).save(f'_f{i:03d}.png')
    enc.stdin.write(out.tobytes())
enc.stdin.close(); enc.wait()
