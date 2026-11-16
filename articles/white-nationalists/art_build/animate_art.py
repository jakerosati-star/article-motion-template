"""White Nationalists art: a three-depth parallax push-in (Jones nearest, the crowd silhouettes, then the banner);
the crowd sways and bobs, each head a little out of step; the stage lights on the banner flicker in the first
second, then a soft spotlight drifts across it and away. Frame 0 = the art (every motion starts from rest).
Run from art_build/ (after cutout.py). Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
art = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
m = np.load('masks.npz'); JON = m['jones'].astype(np.float32); CRW = m['crowd'].astype(np.float32)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

# banner plate: Jones and the crowd removed (inpainted from a clean margin, at full size; only thin bands are seen)
hole = cv2.dilate(((JON + CRW) > 0.02).astype(np.uint8), np.ones((15, 15), np.uint8)) * 255
plate_full = cv2.inpaint(art.astype(np.uint8), hole, 6, cv2.INPAINT_TELEA).astype(np.float32)
near = cv2.dilate(((JON + CRW) > 0.02).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0   # replace only under the layers (+1px)
plate = np.where(near[..., None], plate_full, art)
# layers derived from the art itself, so that frame 0 recombines to exactly the art:
#   crowd: pure black over the banner, so its alpha is how much darker the art is than the rebuilt banner
lum_a = art.mean(2); lum_p = np.maximum(plate.mean(2), 1)
crowd_zone = cv2.dilate((CRW > 0.02).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
CA = np.where(crowd_zone & (JON < 0.02), np.clip(1 - lum_a / lum_p, 0, 1), 0).astype(np.float32)
#   the silhouettes carry on a little way behind Jones, so no banner shows through when he moves against them
under = cv2.dilate((CA > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))) > 0
CA = np.where((JON > 0.02) & under, 1.0, CA).astype(np.float32)
#   Jones: "unmixed" premultiplied colour = art minus what the background behind him contributes at his edges
JC = art - (plate * (1 - CA[..., None])) * (1 - JON[..., None])
sat = plate.max(2) - plate.min(2)
BANNER = cv2.GaussianBlur(np.clip((sat - 25) / 40, 0, 1), (0, 0), 3)   # the lit, coloured banner

def lights(t):
    """Extra light on the banner: a flicker right away, then a spotlight drifting across."""
    fl = 0.0
    for t0, d, a in ((0.2, 0.07, 0.55), (0.33, 0.05, 0.35), (0.5, 0.1, 0.6), (0.75, 0.06, 0.3)):   # stutters
        if t0 <= t < t0 + d: fl = max(fl, a)
    sx = 300 + 900 * (0.5 - 0.5 * math.cos(TAU * t / P))               # sweeps right, then back
    env = math.sin(math.pi * t / P) ** 2                                # 0 at the loop point
    spot = np.exp(-(((xx - sx) / 420) ** 2 + ((yy - 380) / 300) ** 2)) * 0.45 * env
    return (fl * 0.6 + spot) * BANNER

def sway(t):
    """Displacement of the crowd: a slow wave whose phase changes along the crowd, zero at t = 0."""
    w = TAU * t / P
    ph = xx / 170.0
    dx = 9 * (np.sin(2 * w + ph) - np.sin(ph))
    dy = 4 * (np.sin(3 * w + ph * 1.3) - np.sin(ph * 1.3))
    return dx, dy

CX, CY = 0.66 * W, 0.42 * H                                             # toward Jones
def cam(t, depth):
    e = 0.5 - 0.5 * math.cos(TAU * t / P)
    z = 1 + 0.07 * e * depth
    return np.float32([[z, 0, (1 - z) * CX], [0, z, (1 - z) * CY]])
warp = lambda img, M, fl=cv2.INTER_CUBIC: cv2.warpAffine(img, M, (W, H), flags=fl, borderMode=cv2.BORDER_REFLECT)

def frame(t):
    L = lights(t)[..., None]
    bg = 255 - (255 - plate) * (1 - L * np.array([1.0, 0.75, 0.8], np.float32))   # warm pink stage light (screen)
    out = warp(bg, cam(t, 0.3))
    dx, dy = sway(t)
    crowd_a = cv2.remap(CA, xx - dx, yy - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    crowd_a = cv2.warpAffine(crowd_a, cam(t, 0.55), (W, H), flags=cv2.INTER_LINEAR)[..., None]
    out = out * (1 - crowd_a)                                            # the silhouettes are pure black
    Mj = cam(t, 1.0)
    ja = cv2.warpAffine(JON, Mj, (W, H), flags=cv2.INTER_LINEAR)[..., None]
    jc = cv2.warpAffine(JC, Mj, (W, H), flags=cv2.INTER_CUBIC)
    return out * (1 - ja) + jc

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
