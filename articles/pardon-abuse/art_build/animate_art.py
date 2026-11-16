"""Pardon Abuse art: Trump's signature writes itself onto the pardon, left to right, with a pen's uneven pace
(0.25s-1.55s), holds, and the ink fades away before the loop point, so frame 0 = the art without the signature.
The signature is lifted from the still and the paper under it rebuilt by inpainting.
Camera: a parallax push-in. The foreground (people, Trump, the pardon; cutout.py -> fg.npy) pushes in faster than
the paper-and-circle background, and the camera drifts sideways, so the background slides behind them. The
background behind the foreground is rebuilt by inpainting (only a thin band of it is ever seen).
Run from art_build/ (after cutout.py). Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS)
art = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
X0, Y0, X1, Y1 = 1585, 705, 1752, 832                        # signature box on the right-hand page
box = art[Y0:Y1, X0:X1]
lum = box.mean(2)
ink = np.clip((200 - lum) / 120, 0, 1)                       # ink coverage from darkness (paper ~245)
ink[lum > 236] = 0
m = (cv2.dilate((lum < 236).astype(np.uint8), np.ones((7, 7), np.uint8)) * 255)
paper = art.copy()
paper[Y0:Y1, X0:X1] = cv2.inpaint(box.astype(np.uint8), m, 5, cv2.INPAINT_TELEA).astype(np.float32)
paper[Y0:Y1, X0:X1] = cv2.GaussianBlur(paper[Y0:Y1, X0:X1], (0, 0), 1.2) * (m[..., None] > 0) + paper[Y0:Y1, X0:X1] * (m[..., None] == 0)
# the ink's own look: the original pixels where the ink is, over the rebuilt paper
alpha = cv2.GaussianBlur((m > 0).astype(np.float32), (0, 0), 0.8)
sig = box

cols = np.arange(X1 - X0, dtype=np.float32)[None, :]
def pen(t):
    """How far the pen has got (px across the signature): uneven pace, quick strokes with brief slowdowns."""
    s = np.clip((t - 0.25) / 1.3, 0, 1)
    s = s + 0.035 * math.sin(s * math.pi * 7)                # little pauses at each loop of the scrawl
    s = 0.5 - 0.5 * math.cos(math.pi * np.clip(s, 0, 1))     # ease in and out
    return s * (X1 - X0 + 8)

FG = np.load('fg.npy').astype(np.float32)
hole = (cv2.dilate((FG > 0.02).astype(np.uint8), np.ones((5, 5), np.uint8)) * 255)
small = cv2.resize(art.astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_AREA)   # inpaint at half size (fast), then
plate_s = cv2.inpaint(small, cv2.resize(hole, (W // 2, H // 2)), 6, cv2.INPAINT_TELEA)       # use it only under the hole
plate = art.copy(); hm = (hole > 0)[..., None]
plate = np.where(hm, cv2.resize(plate_s, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32), plate)
TAU = 2 * math.pi
CX, CY = 0.56 * W, 0.42 * H                  # push toward Trump and the pardon
def cam(t, depth):
    """Affine for a layer at `depth` (1 = foreground, 0.35 = background): push in and back out, drift sideways."""
    e = 0.5 - 0.5 * math.cos(TAU * t / P)
    z = 1 + 0.075 * e * depth
    dx = 26 * math.sin(TAU * t / P) * depth
    return np.float32([[z, 0, (1 - z) * CX + dx], [0, z, (1 - z) * CY]])

def frame(t):
    f = paper.copy()
    fade = 1 - np.clip((t - 8.6) / 0.7, 0, 1) ** 2           # ink fades out before the loop point
    reveal = np.clip((pen(t) - cols) / 4, 0, 1)              # soft 4px pen front
    a = (alpha * reveal * fade)[..., None]
    f[Y0:Y1, X0:X1] = f[Y0:Y1, X0:X1] * (1 - a) + sig * a
    Mb, Mf = cam(t, 0.35), cam(t, 1.0)
    bg = cv2.warpAffine(plate, Mb, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    fa = cv2.warpAffine(FG, Mf, (W, H), flags=cv2.INTER_LINEAR)[..., None]
    fc = cv2.warpAffine(f * FG[..., None], Mf, (W, H), flags=cv2.INTER_CUBIC)   # premultiplied: no fringe
    return bg * (1 - fa) + fc

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        for t in map(float, sys.argv[1:]): Image.fromarray(np.clip(frame(t), 0, 255).astype(np.uint8)).save(f'_t{t:.2f}.png')
        sys.exit()
    enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                           stdin=subprocess.PIPE)
    for i in range(N): enc.stdin.write(np.clip(frame(i / FPS), 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close(); enc.wait()
