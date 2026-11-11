"""Destroy All Monsters art: the dragon's fire plume licks and shimmers, Trump's tongue flicks (first one in the
first second), the UFO zips around the top third, old monster-movie film flicker + grain + the odd scratch, slow push-in.
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
tongue = poly(TONGUE, 0.8)
ufo = np.load('ufo_alpha.npy')   # AI cut-out (u2net) of the saucer, smoke trail removed, pod kept: see README in this folder
TROOT = (1548.0, 506.0)
# the UFO leaves home for most of the loop: rebuild the sky behind it as soft haze + matching grain
hole_u = cv2.dilate(((ufo > 0.05) | (poly(UFO) > 0.5)).astype(np.uint8), np.ones((11, 11), np.uint8))   # old outline too: the trail goes with it
_p = cv2.inpaint(a.astype(np.uint8), hole_u, 15, cv2.INPAINT_NS).astype(np.float32)
_g = cv2.GaussianBlur(np.random.default_rng(2).normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.0)
_g *= (a - cv2.GaussianBlur(a, (0, 0), 2))[hole_u == 0].std()
plate = np.where(hole_u[..., None] > 0, cv2.GaussianBlur(_p, (0, 0), 8) + _g[..., None], a)
# tongue flick = a local bend (no cut-out, so nothing behind it is ever revealed):
# weight is 1 on the tongue and fades out ~30px around it; bend grows from 0 at the root to full at the tip
tw = ndimage.gaussian_filter(cv2.dilate((tongue > 0.3).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(np.float32), 9)
tu = np.clip((1552 - xx) / (1552 - 1376), 0, 1) ** 1.6
TBEND = (tw * tu).astype(np.float32)
TL = np.dstack([a, tongue * 255]); UL = np.dstack([a, ufo * 255])
ULP = np.dstack([a * ufo[..., None], ufo * 255]).astype(np.float32)   # premultiplied UFO layer

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
UC = (1445.0, 75.0)                       # UFO centre
# (time, dx, dy) waypoints relative to home: repeated points are hovers, the gaps between them are darts
UKEYS = [(0.0, 0, 0), (0.45, 0, 0), (0.85, -700, 120), (1.7, -700, 120), (2.05, -1150, 30), (2.9, -1150, 30),
         (3.25, -300, 170), (4.1, -300, 170), (4.4, 200, 60), (5.2, 200, 60), (5.6, -900, 200), (6.5, -900, 200),
         (6.85, -450, 10), (7.7, -450, 10), (8.15, 0, 0), (10.0, 0, 0)]
def ufo_pos(tt):
    for k in range(1, len(UKEYS)):
        t0, x0, y0 = UKEYS[k - 1]; t1, x1, y1 = UKEYS[k]
        if tt <= t1:
            u = 0.0 if t1 == t0 else max(0.0, min(1.0, (tt - t0) / (t1 - t0)))
            u = u * u * u * (u * (6 * u - 15) + 10)
            return x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
    return 0.0, 0.0
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
    # UFO zips around the top third: quick darts between hovers, banking into each move, motion-blurred when fast
    px, py = ufo_pos(t); qx, qy = ufo_pos(t - 1 / FPS)
    vx, vy = px - qx, py - qy; speed = math.hypot(vx, vy)
    bob = 4 * math.sin(TAU * 7 * t / P) * (1 - min(1, speed / 20))
    bank = max(-14.0, min(14.0, -vx * 0.35))
    sc = 1 + 0.12 * (py / 220.0)
    n_sub = 1 if speed < 12 else 5
    acc = np.zeros((H, W, 4), np.float32)
    for s_ in range(n_sub):
        ox, oy = px - vx * (s_ / n_sub) * 0.8, py - vy * (s_ / n_sub) * 0.8 + bob
        M = cv2.getRotationMatrix2D(UC, bank, sc); M[0, 2] += ox; M[1, 2] += oy
        acc += cv2.warpAffine(ULP, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    w = acc / n_sub; al = w[..., 3] / 255
    f = f * (1 - al[..., None]) + w[..., :3]                     # premultiplied: soft edges and blur stay true to colour
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
    if i in (20, 27, 50, 140, 299): Image.fromarray(out).save(f'_f{i:03d}.png')
    enc.stdin.write(out.tobytes())
enc.stdin.close(); enc.wait()
