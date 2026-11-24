"""Trump / American Communism art: the hammer and sickle swells toward the camera and stamps back down with a red
flash (0.15s, again at 5s); the flag behind ripples gently in a seamless wave; Trump (rembg) stays put on top, with a
slow push-in. The emblem only ever scales up from its place, so nothing behind it is uncovered; the flag ripple uses
a plate with the emblem and Trump inpainted out. Frame 0 = the art. Run from art_build/. Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
art = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
TR = np.load('rb.npy').astype(np.float32) / 255
n_, l_, s_, _ = cv2.connectedComponentsWithStats((TR > 0.5).astype(np.uint8), 8)
TR = TR * cv2.dilate((l_ == 1 + int(np.argmax(s_[1:, 4]))).astype(np.uint8), np.ones((5, 5), np.uint8))
r, g, b = art[..., 0], art[..., 1], art[..., 2]
hs = ((r > 150) & (g < 80) & (b < 80)).astype(np.uint8)
hs = cv2.morphologyEx(hs, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
n, lab, st, cen = cv2.connectedComponentsWithStats(hs, 8)
k = 1 + int(np.argmax(st[1:, 4]))
from scipy import ndimage
HS = ndimage.binary_fill_holes(lab == k).astype(np.float32)
HS = cv2.GaussianBlur(cv2.dilate(HS, np.ones((3, 3), np.uint8)), (0, 0), 1.0) * (1 - TR)
ys, xs = np.where(HS > 0.5); HCX, HCY = float(xs.mean()), float(ys.mean())
hole = cv2.dilate(((HS > 0.02) | (TR > 0.02)).astype(np.uint8), np.ones((11, 11), np.uint8))
small = lambda x: cv2.resize(x, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
plate = cv2.resize(cv2.inpaint(small(art.astype(np.uint8)), (small(hole) > 0).astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA), (W, H)).astype(np.float32)
plate = np.where((hole > 0)[..., None], plate, art)
HSC = art * HS[..., None]; TRC = art * TR[..., None]
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

def stamp(t):
    for t0 in (0.15, 5.0):
        u = (t - t0) / 0.5
        if 0 <= u < 1:
            if u < 0.35: v = u / 0.35; return 1 + 0.14 * math.sin(math.pi / 2 * v), 0.0
            v = (u - 0.35) / 0.65; return 1 + 0.14 * (1 - v) ** 3 * math.cos(2.5 * math.pi * v) ** 2, max(0.0, 1 - v * 2.5)
    return 1.0, 0.0

def flag(t):
    ph = TAU * t / P
    dy = 7 * (np.sin(xx / 210 - 2 * ph) - np.sin(xx / 210)) * np.clip(xx / 400, 0, 1)   # ripple, pinned at left, 0 at t=0
    dx = 3 * (np.sin(yy / 160 - 2 * ph) - np.sin(yy / 160))
    shade = 1 + 0.06 * ((np.cos(xx / 210 - 2 * ph) - np.cos(xx / 210)) * np.clip(xx / 400, 0, 1))[..., None]
    return cv2.remap(art, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) * shade

FG = np.clip(HS + TR, 0, 1)[..., None]
def frame(t):
    f = flag(t) * (1 - FG) + art * FG                                # the flag ripples; emblem and Trump untouched
    s, flash = stamp(t)
    if s > 1:
        M = np.float32([[s, 0, (1 - s) * HCX], [0, s, (1 - s) * HCY]])
        a = cv2.warpAffine(HS, M, (W, H))[..., None]; c = cv2.warpAffine(HSC, M, (W, H), flags=cv2.INTER_CUBIC)
        sh = cv2.GaussianBlur(a[..., 0], (0, 0), 18)[..., None] * min(1, (s - 1) * 5) * 0.5
        f = f * (1 - np.roll(np.roll(sh, 18, 0), 12, 1))
        f = f * (1 - a) + c
        if flash > 0: f = f + a * flash * np.array([70, 25, 25], np.float32)
        f = f * (1 - TR[..., None]) + TRC
    elif flash > 0:
        f = f + HS[..., None] * flash * np.array([70, 25, 25], np.float32)
    e = 0.5 - 0.5 * math.cos(TAU * t / P); z = 1 + 0.045 * e
    Mz = np.float32([[z, 0, (1 - z) * W * 0.5], [0, z, (1 - z) * H * 0.5]])
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
