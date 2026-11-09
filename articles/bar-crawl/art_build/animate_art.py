"""Bar Crawl art: drinks slosh and spill over the rims in turn, a sparkle off Ossoff's smile, the election screen
flickers and LIVE pulses, slow push-in.
Seamless 10s loop; frame 0 is the original. Writes ../art_loop.mp4"""
import math, random, subprocess
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

glasses = [poly([(790, 512), (1072, 512), (930, 700)], 3), ell(1378, 545, 128, 85, 3), poly([(1478, 508), (1752, 508), (1615, 690)], 3)]
glass_all = np.clip(sum(glasses), 0, 1)
# liquid area: inside each bowl but below its sugar rim, so sloshing never pulls the background into the glass
inner = [poly([(808, 528), (1054, 528), (930, 690)], 4), ell(1378, 555, 112, 66, 4), poly([(1496, 524), (1734, 524), (1615, 680)], 4)]
CEN = [(930, 560), (1378, 540), (1615, 555)]
screen = poly([(760, 0), (1850, 0), (1850, 815), (900, 815), (900, 520), (760, 520)], 8) * (1 - glass_all)
live = ell(1680, 573, 75, 26, 3)
LIQ = [a[y - 8:y + 8, x - 8:x + 8].reshape(-1, 3).mean(0) for x, y in [(930, 600), (1378, 560), (1615, 590)]]

def clamp(x): return max(0.0, min(1.0, x))
def ss(x): x = clamp(x); return x * x * (3 - 2 * x)
def wv(k, t, ph=0): return math.sin(TAU * k * t / P + ph)
SLOSH = [(2, 1.0), (3, -0.8), (2, -1.1)]                      # per glass: (cycles per loop, strength); zero at t = 0
def tilt(g, t): k, s = SLOSH[g]; return 0.07 * s * (math.sin(TAU * k * t / P) + 0.35 * math.sin(TAU * 2 * k * t / P))

def arc(x0, y0, x1, y1, bulge=0.0, n=40):
    return [(x0 + (x1 - x0) * u + bulge * math.sin(math.pi * u), y0 + (y1 - y0) * u) for u in np.linspace(0, 1, n)]
coupe_l = [(1378 - 128 * math.cos(th), 545 + 85 * math.sin(th)) for th in np.linspace(-0.45, 1.45, 40)]
coupe_r = [(1378 + 128 * math.cos(th), 545 + 85 * math.sin(th)) for th in np.linspace(-0.45, 1.45, 40)]
PATHS = {(0, 'L'): arc(791, 513, 926, 696), (0, 'R'): arc(1071, 513, 934, 696), (1, 'L'): coupe_l, (1, 'R'): coupe_r,
         (2, 'L'): arc(1479, 509, 1611, 686), (2, 'R'): arc(1751, 509, 1619, 686)}
DRIPS = [(0, 'R', 1.2), (2, 'L', 2.6), (0, 'L', 5.0), (2, 'R', 7.2)]   # (glass, side, start): the two martinis overflow
SPLASHES = [3.8, 6.3]                                          # the tall middle glass splashes instead (its rim is well above the drink)
DRIP_LEN = 1.9
SPARKS = [2.0, 6.1]; SMILE = (991, 232)

def draw_drip(f, g, side, u):
    """an overflow running down the outside of the glass: lip at the rim, a wet trail, a bead at the front"""
    pts = PATHS[(g, side)]; n = len(pts); head = ss(min(u / 0.75, 1)) * (n - 1)
    fade = 1 - clamp((u - 0.75) / 0.25)
    lay = np.zeros((H, W), np.float32)
    hi = int(head)
    for j in range(hi):                                         # uneven, tapering wet trail
        w = (17 - 7 * j / n) * (1 + 0.25 * math.sin(j * 0.9 + g))
        cv2.line(lay, tuple(int(4 * c) for c in pts[j]), tuple(int(4 * c) for c in pts[j + 1]), 0.8, max(3, int(w)), cv2.LINE_AA, 2)
    for j in range(5, hi, 9):                                   # beads left along the trail
        bx, by = pts[j]; cv2.circle(lay, (int(bx * 4), int(by * 4)), 4 * 9, 0.85, -1, cv2.LINE_AA, 2)
    hx, hy = pts[min(hi, n - 1)]
    cv2.ellipse(lay, (int(hx * 4), int((hy + 4) * 4)), (4 * 13, 4 * 16), 0, 0, 360, 0.95, -1, cv2.LINE_AA, 2)   # teardrop head
    cv2.ellipse(lay, (int(pts[0][0] * 4), int(pts[0][1] * 4 - 8)), (4 * 34, 4 * 11), 0, 0, 360, 0.85 * fade, -1, cv2.LINE_AA, 2)  # sheet spilling over the rim
    lay = cv2.GaussianBlur(lay, (0, 0), 1.2) * fade
    col = np.array([214, 186, 255], np.float32)                 # the drink's lavender
    edge = np.clip(lay - cv2.erode(lay, np.ones((5, 5), np.uint8)), 0, 1)
    f[:] = f * (1 - 0.88 * lay[..., None]) + col * 0.88 * lay[..., None]
    f[:] = f * (1 - 0.5 * edge[..., None]) + np.array([150, 105, 230], np.float32) * 0.5 * edge[..., None]   # glassy edge
    shine = cv2.GaussianBlur(np.roll(np.roll(lay, -4, 0), -3, 1) * lay, (0, 0), 1.5)
    f[:] = f + 140 * np.clip(shine - 0.5, 0, 1)[..., None]

yy2, xx2 = np.mgrid[-60:61, -60:61].astype(np.float32)
def sparkle(u):
    """four-point star with a soft core, grows then shrinks over u in 0..1"""
    s = math.sin(math.pi * u); L = 85 * s + 1e-3
    star = np.exp(-(xx2 ** 2) / 4) * np.exp(-(yy2 ** 2) / (2 * (L / 2) ** 2)) + np.exp(-(yy2 ** 2) / 4) * np.exp(-(xx2 ** 2) / (2 * (L / 2) ** 2))
    core = np.exp(-(xx2 ** 2 + yy2 ** 2) / (2 * (6 * s + 1e-3) ** 2))
    bloom = np.exp(-(xx2 ** 2 + yy2 ** 2) / (2 * 18 ** 2)) * 0.45
    return np.clip((star * 0.9 + core + bloom) * s, 0, 1)

enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                       stdin=subprocess.PIPE)
for i in range(N):
    t = i / FPS
    dy = np.zeros((H, W), np.float32)
    for g in range(3):                                          # liquid surface tilts side to side
        cx, cy = CEN[g]; dy += tilt(g, t) * (xx - cx) * inner[g]
    f = cv2.remap(a, xx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fl = 1 + 0.035 * (wv(13, t) * wv(5, t, 1.2) + 0.5 * wv(29, t, 0.3) - 0.5 * math.sin(0.3))
    pulse = 0.5 - 0.5 * math.cos(TAU * 5 * t / P)
    f = f * (1 + (fl - 1) * screen)[..., None] + (70 * pulse * live)[..., None] * np.array([1, 0.85, 1], np.float32)
    for g, side, t0 in DRIPS:
        u = (t - t0) / DRIP_LEN
        if 0 < u < 1: draw_drip(f, g, side, u)
    for t0 in SPLASHES:
        u = (t - t0) / 0.9
        if 0 < u < 1:
            lay = np.zeros((H, W), np.float32); r = random.Random(int(t0 * 10))
            for _ in range(11):
                vx, vy, r0 = r.uniform(-130, 130), r.uniform(-260, -150), r.uniform(4, 9)
                x = 1378 + r.uniform(-60, 60) + vx * u; y = 482 + vy * u + 420 * u * u     # up and back down
                if y < 486: cv2.circle(lay, (int(x * 4), int(y * 4)), int(r0 * 4 * (1 - 0.4 * u)), 0.9, -1, cv2.LINE_AA, 2)
            lay = cv2.GaussianBlur(lay, (0, 0), 1.0)
            f = f * (1 - 0.85 * lay[..., None]) + np.array([225, 205, 255], np.float32) * 0.85 * lay[..., None]
    for t0 in SPARKS:
        u = (t - t0) / 0.7
        if 0 < u < 1:
            sp = sparkle(u); x, y = SMILE
            f[y - 60:y + 61, x - 60:x + 61] += 255 * sp[..., None]
    z = 1 + 0.035 * (0.5 - 0.5 * math.cos(TAU * t / P))
    Mc = np.float32([[z, 0, (1 - z) * W * 0.55], [0, z, (1 - z) * H * 0.5]])
    out = cv2.warpAffine(np.clip(f, 0, 255).astype(np.uint8), Mc, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if i in (0, 45, 63, 78, 120, 299): Image.fromarray(out).save(f'_f{i:03d}.png')
    enc.stdin.write(out.tobytes())
enc.stdin.close(); enc.wait()
