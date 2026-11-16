"""Spies Like Trump art: the eye blinks. One blink at 0.5s, a double blink mid-loop.
The eyelids are built here (dark skin with grain, a lit rim, a lash line and lashes) and slide in from above and
below the frame; the upper lid does most of the travel. Each frame is averaged over sub-frames for motion blur.
Frame 0 = the art (eye open). Run from art_build/. Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); SUB = 5
art = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
rng = np.random.default_rng(7)
xs = np.arange(W, dtype=np.float32); grain = lambda shape, s: rng.normal(0, s, shape).astype(np.float32)

UARC, LARC = 250.0, 150.0                 # how much the lid edges curve (px, middle vs. corners): they wrap the eyeball
def arc(x, upper):
    uu = np.clip((x - W * 0.52) / (W * 0.74), -1.2, 1.2)
    return (UARC if upper else -LARC) * (1 - uu ** 2)

def make_lid(upper):
    """Lid image (tall). Its leading edge is base + arc(x) at row LH; skin extends away from the eye."""
    LH = H + 400
    rows = np.arange(3 * H + 800, dtype=np.float32)[:, None]
    edge = LH + arc(xs, upper)[None, :]
    d = (edge - rows) if upper else (rows - edge)                               # depth into the lid (px)
    a = np.clip(d + 1.0, 0, 1)
    # skin over a curved lid: broad soft highlight along the rim, falling off into shadow; a crease on the upper lid
    v = 14 + 46 * np.exp(-((d - 70) / 75) ** 2) + 8 * np.exp(-((d - 330) / 120) ** 2) * (1 if upper else 0.3)
    if upper: v = v - 16 * np.exp(-((d - 250) / 22) ** 2)
    v = v - 10 * np.clip(1 - d / 14, 0, 1)                                      # dark lash line at the very edge
    lo_f = cv2.GaussianBlur(grain(v.shape, 1.0), (0, 0), 18); lo_f *= 7 / (lo_f.std() + 1e-6)   # blotchy skin tone
    mid_f = cv2.GaussianBlur(grain(v.shape, 1.0), (0, 0), 3); mid_f *= 3 / (mid_f.std() + 1e-6)
    v = v * (1 + 0.012 * lo_f) + mid_f
    v = v * (0.75 + 0.25 * np.exp(-((xs[None, :] - W * 0.5) / (W * 0.55)) ** 2))   # matches the art's vignette
    img = np.repeat(v[..., None], 3, 2) + grain((v.shape[0], W, 1), 5)
    # lashes: thick, long, curved (this is an extreme close-up), fanning out toward the corners
    m = np.zeros(v.shape, np.uint8)
    n = 95 if upper else 70
    for i in range(n):
        x0 = rng.uniform(-150, W + 150)
        y0 = LH + float(arc(np.float32(x0), upper)) + (-4 if upper else 4)
        L = rng.uniform(150, 290) if upper else rng.uniform(70, 150)
        uu = (x0 - W * 0.52) / (W * 0.74)
        ang = rng.normal(0, 0.18) + 0.55 * uu                                  # radians from straight down/up
        curl = rng.uniform(0.15, 0.35) * np.sign(uu if abs(uu) > 0.05 else rng.uniform(-1, 1))
        w0 = rng.uniform(5, 9) if upper else rng.uniform(3, 6)
        pts = []
        for k in range(12):
            s_ = k / 11; a_ = ang + curl * s_ * 2
            pts.append((x0 + math.sin(a_) * L * s_, y0 + (1 if upper else -1) * math.cos(a_) * L * s_ * (1 - 0.3 * s_)))
        for k in range(11):
            th = max(1, int(round(w0 * (1 - k / 11) ** 0.8)))
            cv2.line(m, tuple(int(c) for c in pts[k]), tuple(int(c) for c in pts[k + 1]), 255, th, cv2.LINE_AA)
    lash_a = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 1.0) * 0.95
    rgba = np.dstack([img * (1 - lash_a[..., None]) + 8 * lash_a[..., None], np.maximum(a, lash_a)])
    sh = np.clip(1 - (-d) / 90, 0, 1) * (d < 0) * 0.5                         # shadow the lid casts on the eye
    return rgba, sh, LH

UP, UP_SH, LH = make_lid(True)
LO, LO_SH, _ = make_lid(False)

def place(lid, sh, top):
    t0 = int(math.floor(top)); fr = top - t0
    seg = lid[t0:t0 + H + 1]; s2 = sh[t0:t0 + H + 1]
    return seg[:H] * (1 - fr) + seg[1:H + 1] * fr, s2[:H] * (1 - fr) + s2[1:H + 1] * fr

# lid travel (base of the edge arc): upper from well above the frame (lashes included) down past the middle;
# lower from below the frame up to meet it. Shut, the upper edge (middle 760, corners ~640) overlaps the lower (540 / ~620).
UP_OPEN, UP_SHUT = -560.0, 510.0
LO_OPEN, LO_SHUT = H + 330.0, 700.0
BLINKS = [0.5, 5.0, 5.42]                  # start times (s): a blink in the first second, a double blink mid-loop
def closed(t):
    for t0 in BLINKS:
        s_ = t - t0
        if 0 <= s_ < 0.1: x = s_ / 0.1; return x ** 2                  # fast close, accelerating
        if 0.1 <= s_ < 0.13: return 1.0                                  # one frame shut
        if 0.13 <= s_ < 0.36: x = (s_ - 0.13) / 0.23; return (1 - x) ** 3   # slower open, easing out
    return 0.0

def frame(t):
    c = closed(t)
    if c <= 0: return art
    up_base = UP_OPEN + (UP_SHUT - UP_OPEN) * c; lo_base = LO_OPEN + (LO_SHUT - LO_OPEN) * c
    f = art.copy()
    lo, lsh = place(LO, LO_SH, LH - lo_base)
    up, ush = place(UP, UP_SH, LH - up_base)
    f = f * (1 - np.maximum(ush, lsh)[..., None])
    f = f * (1 - lo[..., 3:]) + lo[..., :3] * lo[..., 3:]
    f = f * (1 - up[..., 3:]) + up[..., :3] * up[..., 3:]
    return f

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:   # test frames: python3 animate_art.py 0.55 0.6 ...
        for t in map(float, sys.argv[1:]): Image.fromarray(np.clip(frame(t), 0, 255).astype(np.uint8)).save(f'_t{t:.2f}.png')
        sys.exit()
    enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                           stdin=subprocess.PIPE)
    for i in range(N):
        t = i / FPS
        if any(0 <= t - b < 0.4 for b in BLINKS):
            f = sum(frame(t + k / (FPS * SUB)) for k in range(SUB)) / SUB   # motion blur over the frame's shutter
        else:
            f = art
        enc.stdin.write(np.clip(f, 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close(); enc.wait()
