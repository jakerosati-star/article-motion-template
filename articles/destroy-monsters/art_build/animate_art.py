"""Destroy All Monsters art: the dragon's fire plume licks and shimmers, Trump's tongue flicks (first one in the
first second), the UFO hovers, old monster-movie film flicker + grain + the odd scratch, slow push-in.
Seamless 10s loop; frame 0 is the original. Writes ../art_loop.mp4"""
import math, random, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw
from scipy import ndimage

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
def poly(pts, blur=0.0):
    m = Image.new('L', (W, H), 0); ImageDraw.Draw(m).polygon(pts, fill=255)
    m = np.array(m, np.float32) / 255
    return ndimage.gaussian_filter(m, blur) if blur else m

# --- tongue + UFO cut out; holes filled ---
TONGUE = [(1556, 486), (1520, 494), (1480, 499), (1450, 506), (1420, 511), (1395, 517), (1377, 527), (1374, 546),
          (1396, 549), (1426, 541), (1456, 529), (1490, 527), (1530, 524), (1558, 516)]
UFO = [(1290, 22), (1340, 8), (1460, 8), (1540, 38), (1592, 92), (1603, 116), (1596, 132), (1500, 147), (1420, 142),
       (1350, 112), (1298, 62)]
tongue = poly(TONGUE, 0.8); ufo = poly(UFO, 0.8)
TROOT = (1548.0, 506.0)
hole_u = cv2.dilate((ufo > 0.05).astype(np.uint8), np.ones((9, 9), np.uint8))
plate = cv2.inpaint(a.astype(np.uint8), hole_u, 9, cv2.INPAINT_TELEA).astype(np.float32)
# tongue flick = a local bend (no cut-out, so nothing behind it is ever revealed):
# weight is 1 on the tongue and fades out ~30px around it; bend grows from 0 at the root to full at the tip
tw = ndimage.gaussian_filter(cv2.dilate((tongue > 0.3).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(np.float32), 9)
tu = np.clip((1552 - xx) / (1552 - 1376), 0, 1) ** 1.6
TBEND = (tw * tu).astype(np.float32)
TL = np.dstack([a, tongue * 255]); UL = np.dstack([a, ufo * 255])

# --- fire plume region: only bright flame pixels move (the dark falling tower stays put) ---
plume = poly([(985, 330), (1060, 300), (1130, 380), (1220, 470), (1300, 600), (1335, 760), (1320, 885), (1180, 892),
              (1100, 760), (1040, 600), (990, 450)], 22)
miller = np.zeros((H, W), np.float32); cv2.ellipse(miller, (1240, 250), (105, 150), 0, 0, 360, 1.0, -1)
plume *= 1 - ndimage.gaussian_filter(miller, 12)
lum = cv2.GaussianBlur(a.mean(2), (0, 0), 3)
bright = np.clip((lum - 150) / 60, 0, 1)
fire_w = (plume * bright).astype(np.float32)
U = np.array([0.45, 0.89], np.float32); U /= np.linalg.norm(U); NRM = np.array([-U[1], U[0]])
along = (xx - 1000) * U[0] + (yy - 320) * U[1]; across = (xx - 1000) * NRM[0] + (yy - 320) * NRM[1]

def ss(x): x = max(0.0, min(1.0, x)); return x * x * (3 - 2 * x)
def wv(k, t, ph=0): return math.sin(TAU * k * t / P + ph)
FLICKS = [0.25, 3.4, 6.6]                 # tongue flicks: a quick lick, two wiggles, back
def tongue_angle(t):
    for t0 in FLICKS:
        u = (t - t0) / 0.7
        if 0 <= u <= 1: return 7.0 * math.sin(math.pi * u) * math.sin(TAU * 2.0 * u + 0.6)
    return 0.0
rng = random.Random(9)
SCRATCH = {i: (rng.uniform(100, 1820), rng.uniform(0.25, 0.5)) for i in rng.sample(range(2, 300), 14)}

enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                       stdin=subprocess.PIPE)
for i in range(N):
    t = i / FPS
    if i == 0: enc.stdin.write(a.astype(np.uint8).tobytes()); continue
    # fire: downstream-travelling ripples, displacement weighted to the bright flame only
    ph = TAU * 6 * t / P
    d_al = 5.0 * np.sin(along * 0.035 - ph) + 2.5 * np.sin(along * 0.08 + across * 0.03 - 1.7 * ph)
    d_ac = 4.0 * np.sin(along * 0.05 - 1.3 * ph + across * 0.02)
    dx = (d_al * U[0] + d_ac * NRM[0]) * fire_w; dy = (d_al * U[1] + d_ac * NRM[1]) * fire_w
    f = cv2.remap(plate, xx - dx, yy - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    f = f * (1 + 0.06 * (0.6 * wv(13, t) + 0.4 * wv(7, t, 1.1)) * fire_w)[..., None]
    # UFO hover
    ux, uy = 6 * wv(1, t, 0.5) - 6 * math.sin(0.5), 4 * wv(3, t)
    w = cv2.warpAffine(UL, np.float32([[1, 0, ux], [0, 1, uy]]), (W, H), flags=cv2.INTER_LINEAR, borderValue=0); al = w[..., 3] / 255
    f = f * (1 - al[..., None]) + w[..., :3] * al[..., None]
    # tongue flick: bend the tip up/down (applied to everything drawn so far, locally)
    ang = tongue_angle(t)
    if abs(ang) > 0.01:
        bend = (ang / 7.0) * 16.0 * TBEND
        f = cv2.remap(f, xx, yy + bend, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # old film: exposure flicker, grain, the odd scratch
    f = f * (1 + 0.035 * wv(19, t) * wv(7, t, 0.4) + 0.02 * wv(31, t, 1.0))
    g = cv2.resize(np.random.default_rng(i).normal(0, 6, (H // 2, W // 2)).astype(np.float32), (W, H))
    f = f + g[..., None]
    if i in SCRATCH:
        sx, op = SCRATCH[i]; cv2.line(f, (int(sx), 0), (int(sx + 6), H), (235, 255, 220), 1, cv2.LINE_AA)
        f = f * 1.0
    z = 1 + 0.035 * (0.5 - 0.5 * math.cos(TAU * t / P))
    Mc = np.float32([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H / 2]])
    out = cv2.warpAffine(np.clip(f, 0, 255).astype(np.uint8), Mc, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if i in (12, 15, 60, 150): Image.fromarray(out).save(f'_f{i:03d}.png')
    enc.stdin.write(out.tobytes())
enc.stdin.close(); enc.wait()
