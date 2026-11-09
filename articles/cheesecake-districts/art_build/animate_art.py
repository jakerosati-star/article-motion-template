"""Cheesecake Districts art: the Cheesecake Factory pin spins like a coin (eased in and out) twice per loop, the flag
pin's metal catches a glint, slow push-in. Suit fabric behind the pin is rebuilt (smooth fill + the suit's vertical
rib texture). Seamless 10s loop; frame 0 is the original. Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image
from scipy import ndimage

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
cx, cy, rx, ry, ang = np.load('pin_ellipse.npy')       # fitEllipse: rx along the 89.5deg axis -> vertical
A, B = ry + 5, rx + 5                                   # horizontal / vertical semi-axes (+ rim)
pin = np.zeros((H, W), np.uint8); cv2.ellipse(pin, (int(cx), int(cy)), (int(A), int(B)), 0, 0, 360, 1, -1, cv2.LINE_AA)
pin_a = ndimage.gaussian_filter(pin.astype(np.float32), 1.0)
x0, x1, y0, y1 = int(cx - A - 40), int(cx + A + 40), int(cy - B - 40), int(cy + B + 40)

# suit behind the pin: smooth inpaint + vertical rib texture tiled down from a strip just below the pin
hole = cv2.dilate(pin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)))   # pin + its shadow ring
lo = cv2.GaussianBlur(a, (0, 0), 6)
smooth = cv2.inpaint(np.clip(lo, 0, 255).astype(np.uint8), hole, 40, cv2.INPAINT_NS).astype(np.float32)
smooth = cv2.GaussianBlur(smooth, (0, 0), 12)
hp = a - cv2.GaussianBlur(a, (0, 0), 6)
strip = hp[y1 + 2:y1 + 62]
tile = np.concatenate([strip, strip[::-1]] * 8)[: y1 - y0]
tex = np.zeros_like(a); tex[y0:y1] = tile[: y1 - y0]
bg = np.where(hole[..., None] > 0, smooth + tex, a)
soft = ndimage.gaussian_filter(hole.astype(np.float32), 8)[..., None]
bg = a * (1 - soft) + bg * soft

face = a[y0:y1, x0:x1].copy(); fa = pin_a[y0:y1, x0:x1].copy()
# back of the pin: brushed gold metal
g = cv2.cvtColor(face.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
g = cv2.GaussianBlur(g, (0, 0), 2.5); g = (g - g.min()) / (g.max() - g.min() + 1e-6)
back = np.dstack([150 + 90 * g, 115 + 70 * g, 45 + 30 * g]).astype(np.float32)
lx, ly = cx - x0, cy - y0

def ss(x): x = max(0.0, min(1.0, x)); return x * x * (3 - 2 * x)
def ease_io(u): return 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, u)))
SPINS = [(0.1, 1.5, 2), (5.0, 1.6, 3)]       # (start, length, full turns)
def theta(t):
    th = 0.0
    for t0, d, n in SPINS: th += TAU * n * ease_io((t - t0) / d)
    return th
def wv(k, t, ph=0): return math.sin(TAU * k * t / P + ph)

enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                       stdin=subprocess.PIPE)
for i in range(N):
    t = i / FPS
    if i == 0: enc.stdin.write(a.astype(np.uint8).tobytes()); continue
    th = theta(t); c = math.cos(th); sx = max(abs(c), 0.04)
    f = bg.copy()
    src = face if c >= 0 else back
    M = np.float32([[sx * (1 if c >= 0 else -1), 0, lx - sx * (1 if c >= 0 else -1) * lx], [0, 1, 0]])
    w = cv2.warpAffine(src, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
    al = cv2.warpAffine(fa, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
    shade = 0.75 + 0.25 * abs(c)                                   # darker when edge-on
    spec = 70 * max(0.0, math.sin(th * 2)) ** 6 * (abs(c) > 0.2)    # flash as it turns toward the light
    w = np.clip(w * shade + spec, 0, 255)
    sh = cv2.GaussianBlur(al, (0, 0), 6) * 0.35                     # contact shadow on the suit
    reg = f[y0:y1, x0:x1]
    reg = reg * (1 - np.roll(np.roll(sh, 6, 0), 4, 1)[..., None]); reg = reg * (1 - al[..., None]) + w * al[..., None]
    f[y0:y1, x0:x1] = reg
    z = 1 + 0.035 * (0.5 - 0.5 * math.cos(TAU * t / P))
    Mc = np.float32([[z, 0, (1 - z) * cx], [0, z, (1 - z) * cy]])
    out = cv2.warpAffine(np.clip(f, 0, 255).astype(np.uint8), Mc, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if i in (12, 20, 27, 160, 299): Image.fromarray(out).save(f'_f{i:03d}.png')
    enc.stdin.write(out.tobytes())
enc.stdin.close(); enc.wait()
