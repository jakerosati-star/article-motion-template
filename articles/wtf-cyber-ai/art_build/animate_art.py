"""WTF Is Happening art: a digital glitch. In bursts (0.1s, 1.4s, 4.6s, 7.3s), horizontal bands of the frame tear
sideways with a red/cyan channel split and a few flashes of blocky noise; between bursts the image holds, with a
faint scanline shimmer and the outage panel's text blinking. A slow push-in throughout. Frame 0 = the art, and
the glitches are seeded, so every render is identical. Run from art_build/. Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
art = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
BURSTS = [(0.1, 0.5), (1.4, 0.25), (4.6, 0.45), (7.3, 0.3)]    # (start, length)
PANEL = (1205, 660, 1920, 980)                                 # the outage notice

def glitch_amount(t):
    for t0, d in BURSTS:
        if t0 <= t < t0 + d:
            u = (t - t0) / d; return math.sin(math.pi * u) ** 0.5
    return 0.0

def frame(t):
    i = int(round(t * FPS)); rng = np.random.default_rng(1000 + i)
    f = art.copy()
    g = glitch_amount(t)
    if g > 0:
        # tearing bands
        for _ in range(int(2 + 5 * g)):
            y0 = int(rng.integers(0, H - 20)); h = int(rng.integers(8, 90)); dx = int(rng.normal(0, 45 * g))
            f[y0:y0 + h] = np.roll(f[y0:y0 + h], dx, 1)
        # channel split
        s = int(3 + 8 * g)
        f[..., 0] = np.roll(f[..., 0], s, 1); f[..., 2] = np.roll(f[..., 2], -s, 1)
        # a few blocks of noise / inverted blocks
        for _ in range(int(1 + 3 * g)):
            w = int(rng.integers(40, 260)); h = int(rng.integers(10, 50)); x0 = int(rng.integers(0, W - w)); y0 = int(rng.integers(0, H - h))
            if rng.random() < 0.5: f[y0:y0 + h, x0:x0 + w] = 255 - f[y0:y0 + h, x0:x0 + w]
            else: f[y0:y0 + h, x0:x0 + w] = rng.integers(0, 2, (h, w, 1)) * np.array([230, 20, 30], np.float32)
    # outage panel text blinks (dims) a few times
    x0, y0, x1, y1 = PANEL
    blink = 1.0 if (int(t * 3) % 4 == 3 and t > 1.0) else 0.0
    if blink and t < P - 0.4:
        txt = f[y0:y1, x0:x1]; lum = txt.mean(2, keepdims=True)
        f[y0:y1, x0:x1] = np.where(lum > 120, txt * 0.35, txt)
    # scanline shimmer
    f *= (1 - 0.035 * (0.5 + 0.5 * np.sin(np.arange(H)[:, None, None] * 0.9 + t * 40))) if t > 0 else 1
    e = 0.5 - 0.5 * math.cos(TAU * t / P); z = 1 + 0.04 * e
    M = np.float32([[z, 0, (1 - z) * W * 0.5], [0, z, (1 - z) * H * 0.5]])
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
