"""Axis of Evil art: every other grid tile flips over in turn, in a diagonal wave from the top left (each tile turns on
its vertical axis, darkening as it turns edge-on, over a dark ember backing), at 0.15s and again at 5s; a slow
push-in throughout. Frame 0 = the art. Grid lines measured from the art. Run from art_build/. Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
art = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
XS = [0, 105, 372, 644, 911, 1173, 1439, 1706, W]      # grid line centres (and the frame edges)
YS = [0, 174, 418, 660, 905, H]
TILES = [(XS[i], YS[j], XS[i + 1], YS[j + 1]) for j in range(len(YS) - 1) for i in range(len(XS) - 1)]
BACK = np.array([38, 14, 6], np.float32)               # what shows behind a turning tile
WAVES = [0.15, 5.0]; DUR = 0.36

def flip(k, t):
    """0 = flat (normal), 1 = edge-on"""
    x0, y0, x1, y1 = TILES[k]
    ci, cj = XS.index(x0), YS.index(y0)
    if (ci + cj) % 2: return 0.0                                     # every other tile, checkerboard
    delay = 0.085 * (ci + cj)
    for w in WAVES:
        u = (t - w - delay) / DUR
        if 0 <= u < 1: return math.sin(math.pi * u)
    return 0.0

def frame(t):
    f = art.copy()
    for k, (x0, y0, x1, y1) in enumerate(TILES):
        a = flip(k, t)
        if a <= 0: continue
        tile = art[y0:y1, x0:x1]; tw = x1 - x0
        sx = max(1, int(round(tw * (1 - a))))                      # cos-like squeeze toward the tile's centre
        sq = cv2.resize(tile, (sx, y1 - y0), interpolation=cv2.INTER_AREA)
        sq = sq * (1 - 0.55 * a) + np.array([255, 150, 60], np.float32) * 0.18 * a * (1 - a) * 4   # darkens, rim of light
        block = np.empty_like(tile); block[:] = BACK
        o = (tw - sx) // 2; block[:, o:o + sx] = sq
        f[y0:y1, x0:x1] = block
    e = 0.5 - 0.5 * math.cos(TAU * t / P); z = 1 + 0.045 * e
    M = np.float32([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H * 0.42]])
    return cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        for t in map(float, sys.argv[1:]): Image.fromarray(np.clip(frame(t), 0, 255).astype(np.uint8)).save(f'_t{t:.2f}.png')
        sys.exit()
    enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'], stdin=subprocess.PIPE)
    for i in range(N): enc.stdin.write(np.clip(frame(i / FPS), 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close(); enc.wait()
