"""Vance Focus Groups art: Vance sours: his brows pinch down and in, his eyes narrow and his mouth tightens into a
grimace (0.15s, again at 5s), easing back each time; beside him the other man's laugh bounces (head and shoulders
bob in quick chuckles). All smooth local warps of the art (no cut-outs, nothing uncovered); slow push-in on Vance.
Frame 0 = the art; everything is at rest by the loop point. Run from art_build/. Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
art = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
def bump(cx, cy, rx, ry, p=2):
    return np.clip(1 - ((xx - cx) / rx) ** 2 - ((yy - cy) / ry) ** 2, 0, 1) ** p

BROW_L, BROW_R = bump(755, 258, 70, 34), bump(858, 255, 70, 34)
EYE_L, EYE_R = bump(715, 286, 62, 30), bump(868, 282, 62, 30)
MOUTH = bump(790, 410, 95, 45)
CORN_L, CORN_R = bump(738, 412, 40, 34), bump(845, 410, 40, 34)
HEAD2 = bump(1412, 360, 200, 250, 1.5)          # the laughing man's head
BODY2 = bump(1420, 720, 330, 330, 1.5)          # and shoulders

def ease(u): u = min(1, max(0, u)); return u * u * (3 - 2 * u)
def wince(t):
    v = 0.0
    for t0 in (0.15, 5.0):
        v = max(v, ease((t - t0) / 0.35) * (1 - ease((t - t0 - 1.6) / 0.9)))
    return v
def laugh(t):
    v = 0.0
    for t0, n in ((0.3, 4), (2.9, 3), (6.2, 4)):
        u = t - t0
        if 0 <= u < n * 0.24:
            v = math.sin(math.pi * (u / 0.24 % 1)) * (1 - u / (n * 0.24)) ** 0.5
    return v

def frame(t):
    w = wince(t); l = laugh(t)
    dx = np.zeros((H, W), np.float32); dy = np.zeros((H, W), np.float32)
    # brows pinch: inner ends down and toward the nose
    dx += w * (7 * BROW_L - 7 * BROW_R); dy += w * 8 * (BROW_L + BROW_R)
    # eyes narrow
    dy += w * 0.22 * ((yy - 286) * EYE_L + (yy - 282) * EYE_R)
    # mouth tightens: lips press together, corners pull down and in
    dy += w * 0.18 * (yy - 410) * MOUTH
    dy += w * 7 * (CORN_L + CORN_R); dx += w * (4 * CORN_L - 4 * CORN_R)
    # laugh: head bobs and tips back a touch, shoulders follow
    dy += -l * 10 * HEAD2 - l * 5 * BODY2
    dx += l * 3 * HEAD2
    f = cv2.remap(art, xx - dx, yy - dy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    e = 0.5 - 0.5 * math.cos(TAU * t / P); z = 1 + 0.05 * e
    Mz = np.float32([[z, 0, (1 - z) * 800], [0, z, (1 - z) * 330]])
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
