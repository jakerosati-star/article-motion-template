"""Miller 2028 art: the red marker scribbles crossing out Vance (left) and Rubio (right) get re-traced: a hot pen
tip runs along each stroke in drawing order, the ink behind it turning bolder, then settling (0.1s, again at 5s);
slow push-in on Miller. The scribbles never come off (the faces under them can't be rebuilt cleanly), so frame 0
is the art and nothing underneath is ever exposed. Run from art_build/. Writes ../art_loop.mp4"""
import math, subprocess
from collections import deque
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
art = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
r, g, b = art[..., 0], art[..., 1], art[..., 2]
xx = np.arange(W)[None, :].repeat(H, 0)
red = np.clip((r - np.maximum(g, b) - 40) / 60, 0, 1) * ((xx < 830) | (xx > 1130))
m = (red > 0.3).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
comps = sorted([i for i in range(1, n) if st[i, 4] > 1000], key=lambda i: st[i, 0])
ALPHA = cv2.GaussianBlur(np.clip(red * 1.3, 0, 1), (0, 0), 0.7) * (cv2.dilate(np.isin(lab, comps).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)

# geodesic distance along each scribble from its start (half resolution BFS)
S = 2
mh = cv2.resize(np.isin(lab, comps).astype(np.uint8), (W // S, H // S), interpolation=cv2.INTER_NEAREST)
mh = cv2.dilate(mh, np.ones((3, 3), np.uint8))
labh = cv2.resize(lab, (W // S, H // S), interpolation=cv2.INTER_NEAREST)
DIST = np.full(mh.shape, -1.0, np.float32); FRAC = np.zeros(mh.shape, np.float32); WHICH = np.full(mh.shape, -1, np.int32)
nh, lh, sh, _ = cv2.connectedComponentsWithStats(mh, 8)
pieces = sorted([i for i in range(1, nh) if sh[i, 4] > 150], key=lambda i: sh[i, 0])
for k, i in enumerate(pieces):
    ys, xs = np.where(lh == i); j = int(np.argmin(xs)); start = (ys[j], xs[j])
    d = np.full(mh.shape, -1, np.int32); d[start] = 0; q = deque([start])
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            yy_, xx_ = y + dy, x + dx
            if 0 <= yy_ < mh.shape[0] and 0 <= xx_ < mh.shape[1] and lh[yy_, xx_] == i and d[yy_, xx_] < 0:
                d[yy_, xx_] = d[y, x] + 1; q.append((yy_, xx_))
    sel = d >= 0
    FRAC[sel] = d[sel] / d.max(); WHICH[sel] = k
FRAC = cv2.resize(FRAC, (W, H), interpolation=cv2.INTER_LINEAR)
WHICH = cv2.resize(WHICH.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
cx = [sh[i, 0] + sh[i, 2] / 2 for i in pieces]
# timing: pieces on the left side (Vance) first, then the right (Rubio)
START = [0.1 if c < W / 2 else 0.75 + 0.35 * sum(1 for c2 in cx[:k] if c2 >= W / 2) for k, c in enumerate(cx)]
DRAW = [0.6 if c < W / 2 else 0.55 for c in cx]

# the photo under the scribbles is black and white: rebuild it in grey (no colour can bleed in), smooth fill from
# both inpainting methods averaged, then the photo's own film grain laid back over it so it isn't flat
hole = cv2.dilate((ALPHA > 0.02).astype(np.uint8), np.ones((11, 11), np.uint8))
gray = art.mean(2).astype(np.uint8)
f1 = cv2.inpaint(gray, hole * 255, 5, cv2.INPAINT_TELEA).astype(np.float32)
f2 = cv2.inpaint(gray, hole * 255, 5, cv2.INPAINT_NS).astype(np.float32)
fill = cv2.GaussianBlur((f1 + f2) / 2, (0, 0), 2.0)
known = (hole == 0) & (np.abs(art[..., 0] - art[..., 2]) < 12)
grain_src = (art.mean(2) - cv2.GaussianBlur(art.mean(2), (0, 0), 1.5))
ys_, xs_ = np.where(known)
yy_, xx_ = np.mgrid[0:H, 0:W]
idx = (yy_.astype(np.int64) * 7919 + xx_.astype(np.int64) * 104729) % len(ys_)
fill = fill + grain_src[ys_[idx], xs_[idx]]
plate = np.where((hole > 0)[..., None], np.repeat(fill[..., None], 3, 2), art)
fe = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 2)[..., None]
plate = art * (1 - fe) + plate * fe

# the scribbles never come off (the faces under them can't be rebuilt cleanly): instead the marker RE-TRACES them.
# A hot, bright pen front runs along each stroke in drawing order (Vance first, then Rubio), and the ink it has
# passed swells a touch bolder and glossier, then settles. Again at 5s. Only red pixels and a thin rim beyond
# them change, so nothing underneath is ever exposed.
INK = np.array([235, 20, 30], np.float32)
BOLD = cv2.dilate((ALPHA > 0.3).astype(np.uint8), np.ones((11, 11), np.uint8)).astype(np.float32)
BOLD = cv2.GaussianBlur(BOLD, (0, 0), 1.0)
PASSES = [0.0, 4.9]
def frame(t):
    front = np.zeros((H, W), np.float32); passed = np.zeros((H, W), np.float32)
    for t0 in PASSES:
        for k in range(len(pieces)):
            u = (t - t0 - START[k]) / DRAW[k]
            if u < 0 or u > 1.8: continue
            p = min(1.0, 1 - (1 - min(u, 1)) ** 1.6)
            sel = WHICH == k
            d = p * 1.03 - FRAC[sel]
            front[sel] = np.maximum(front[sel], np.exp(-(d / 0.045) ** 2) * (1 if u <= 1 else max(0, 1.8 - u) / 0.8))
            fade = 1 if u <= 1 else max(0.0, 1 - (u - 1) / 0.8)
            passed[sel] = np.maximum(passed[sel], np.clip(d / 0.03, 0, 1) * fade)
    f = art.copy()
    bold = (BOLD * passed * 0.95)[..., None]
    f = f * (1 - bold) + INK * bold                                   # bolder ink behind the pen
    glow = cv2.GaussianBlur(front * BOLD, (0, 0), 6)[..., None]
    hot = (front * BOLD)[..., None]
    f = 255 - (255 - f) * (1 - np.clip(1.6 * glow, 0, 0.9) * np.array([1, 0.6, 0.5], np.float32))   # warm glow around the pen tip
    f = f * (1 - hot) + np.array([255, 235, 210], np.float32) * hot                 # hot pen tip
    e = 0.5 - 0.5 * math.cos(TAU * t / P); z = 1 + 0.05 * e
    Mz = np.float32([[z, 0, (1 - z) * 960], [0, z, (1 - z) * 400]])
    return cv2.warpAffine(f, Mz, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        for t in map(float, sys.argv[1:]): Image.fromarray(np.clip(frame(t), 0, 255).astype(np.uint8)).save(f'_t{t:.2f}.png')
        sys.exit()
    enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'], stdin=subprocess.PIPE)
    for i in range(N): enc.stdin.write(np.clip(frame(i / FPS), 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close(); enc.wait()
