"""Two Biggest Rats art: the Vance and Musk rats get jumpy. They twitch, then each makes quick nervous darts away
from the other and back (Vance left, Musk right), with a little hop, right away and again mid-loop; Trump's glare
holds with a slow push-in. Short darts only, so just thin slivers behind the rats are ever revealed (rebuilt by
inpainting; the ledge row by row). Rats = rembg body + tails + feet and their ledge shadows (extra.npy).
Frame 0 = the art. Run from art_build/. Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
art = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
rb = np.load('rb.npy').astype(np.float32) / 255
n, lab, st, cen = cv2.connectedComponentsWithStats((rb > 0.5).astype(np.uint8), 8)
ids = sorted([i for i in range(1, n) if st[i, 4] > 5000], key=lambda i: cen[i][0])
RATS = [cv2.dilate((lab == i).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32) * rb for i in ids]
extra = np.load('extra.npy')
mid = (cen[ids[0]][0] + cen[ids[1]][0]) / 2
xx_ = np.arange(W)[None, :].repeat(H, 0)
for k, side in enumerate((xx_ < mid, xx_ >= mid)):
    RATS[k] = np.maximum(RATS[k], cv2.GaussianBlur((extra & side).astype(np.float32), (0, 0), 0.8))
CEN = [tuple(cen[i]) for i in ids]
ALL = np.clip(sum(RATS), 0, 1)
# ledge: the white strip at the bottom; find its top edge
rowL = art.mean(2).mean(1); LEDGE = int(np.argmax(rowL[700:] > 200)) + 700
ALL = np.clip(sum(RATS), 0, 1)
hole = cv2.dilate((ALL > 0.02).astype(np.uint8), np.ones((9, 9), np.uint8))
plate = cv2.inpaint(art.astype(np.uint8), hole * 255, 9, cv2.INPAINT_TELEA).astype(np.float32)
for y in range(LEDGE - 1, H):                                   # the ledge: row by row from its uncovered pixels
    free = hole[y] == 0
    if free.any(): plate[y, hole[y] > 0] = np.median(art[y, free], 0)
plate = np.where((hole > 0)[..., None], plate, art)
# contact shadow under each rat, so they sit on the ledge as they move
SHADOWS = []
for m in RATS:
    ys, xs = np.where(m > 0.5); sh = np.zeros((H, W), np.float32)
    cv2.ellipse(sh, (int(xs.mean()), int(LEDGE + 30)), (int((xs.max() - xs.min()) * 0.4), 14), 0, 0, 360, 1, -1)
    SHADOWS.append(cv2.GaussianBlur(sh, (0, 0), 8) * 0.0)   # the art's own shadow travels with the rat layer

def ease_in_out(u): return 0.5 - 0.5 * math.cos(math.pi * min(1, max(0, u)))
DARTS = [0.35, 0.95, 5.0, 5.55]          # each: a quick dart out (0.18s), a beat, and back (0.25s)
def rat_dx(k, t):
    side = -1 if k == 0 else 1
    tw = 5 * math.sin(TAU * 8 * t) * math.exp(-((t - 0.15) / 0.1) ** 2)               # nervous twitch
    d = 0.0
    for j, t0 in enumerate(DARTS):
        t1 = t0 + 0.05 * k
        out = ease_in_out((t - t1) / 0.18); back = ease_in_out((t - t1 - 0.3) / 0.25)
        d = max(d, (55 if j % 2 == 0 else 35) * out * (1 - back))
    return side * d + tw
def rat_hop(k, t):
    for t0 in DARTS:
        u = (t - t0 - 0.05 * k) / 0.18
        if 0 <= u < 1: return -18 * math.sin(math.pi * u)
    return 0.0

def layers(t, rest=False):
    f = plate.copy()
    for k, m in enumerate(RATS):
        dx, dy = (0.0, 0.0) if rest else (rat_dx(k, t), rat_hop(k, t))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        a = cv2.warpAffine(m, M, (W, H))[..., None]; c = cv2.warpAffine(art * m[..., None], M, (W, H), flags=cv2.INTER_CUBIC)
        f = f * (1 - a) + c
    return f

REST = None
def frame(t):
    global REST
    if REST is None: REST = layers(0, rest=True)
    f = art + layers(t) - REST                                # frame 0 is exactly the art
    e = 0.5 - 0.5 * math.cos(TAU * t / P); z = 1 + 0.05 * e
    Mz = np.float32([[z, 0, (1 - z) * W * 0.5], [0, z, (1 - z) * H * 0.35]])
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
