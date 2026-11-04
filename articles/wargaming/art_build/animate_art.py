"""Wargaming art: two hands float independently over the blurred suit, glint across the ballot box sign.
Seamless 10s loop, full image on screen from frame 0. Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw
from scipy import ndimage

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
m = np.array(Image.open('rembg_mask.png')).astype(np.float32) / 255

# foreground (hands, sleeves, agent, card): solid inside, soft outline
hard = ndimage.binary_fill_holes(m > 0.5)
fg_a = np.where(ndimage.binary_erosion(hard, iterations=2), 1.0, m)
SPLIT = 840                                   # left group (agent + left hand) | right group (card + right hand)
left_a = fg_a.copy(); left_a[:, SPLIT:] = 0
right_a = fg_a.copy(); right_a[:, :SPLIT] = 0

# background: blurred suit; fill behind the foreground at low res (it's out of focus anyway)
hole = cv2.dilate((fg_a > 0.05).astype(np.uint8), np.ones((25, 25), np.uint8))
small = cv2.resize(a.astype(np.uint8), (480, 270), interpolation=cv2.INTER_AREA)
hs = cv2.resize(hole, (480, 270), interpolation=cv2.INTER_NEAREST)
fill = cv2.inpaint(small, hs, 12, cv2.INPAINT_TELEA)
fill = cv2.GaussianBlur(cv2.resize(fill, (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), 6).astype(np.float32)
grain = np.random.default_rng(2).normal(0, 6, a.shape).astype(np.float32)   # match the image's film grain
bg = np.where(hole[..., None] > 0, np.clip(fill + grain, 0, 255), a)

# card area for the glint (excluding the thumb: skin is red-dominant, card is yellow/white/blue)
cm = Image.new('L', (W, H), 0); ImageDraw.Draw(cm).polygon([(1005, 186), (1594, 448), (1357, 902), (845, 691)], fill=255)
card = (np.array(cm) > 0) & ~((a[..., 0] - a[..., 1]) > 35)
card_a = ndimage.gaussian_filter(card.astype(np.float32), 1.0) * right_a

def rgba(alpha, extra=None):
    out = np.dstack([a, alpha * 255]).astype(np.float32)
    return out if extra is None else np.dstack([out, extra * 255])

L_RGBA = rgba(left_a); R_RGBA = rgba(right_a, card_a)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); diag = xx * 0.75 + yy * 0.66

def wv(k, t, ph=0): return math.sin(TAU * k * t / P + ph)
def clamp(x): return max(0.0, min(1.0, x))
def warp(img, ang_deg, pivot, dx, dy, border=cv2.BORDER_REPLICATE):
    M = cv2.getRotationMatrix2D(pivot, ang_deg, 1.0); M[0, 2] += dx; M[1, 2] += dy
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=border)
def over(base, rgb, al): return base * (1 - al[..., None]) + rgb * al[..., None]

enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                       stdin=subprocess.PIPE)
for i in range(N):
    t = i / FPS
    # background drifts a touch the other way (depth)
    zb = 1.012 + 0.008 * wv(1, t)
    Mb = np.array([[zb, 0, (1 - zb) * W / 2 - 6 * wv(1, t, 0.8)], [0, zb, (1 - zb) * H / 2]], np.float32)
    f = cv2.warpAffine(bg, Mb, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # left hand (+ agent): pivots from an off-screen elbow, bottom left
    L = warp(L_RGBA, 1.1 * wv(1, t, 0.4) + 0.4 * wv(3, t), (-250, 1350), 5 * wv(2, t, 1.1), 7 * wv(1, t, 2.0))
    f = over(f, L[..., :3], L[..., 3] / 255)
    # right hand (+ card): its own rhythm, pivots from off-screen bottom right
    R = warp(R_RGBA, -1.3 * wv(1, t, 2.6) - 0.4 * wv(2, t, 0.9), (2200, 1350), -5 * wv(2, t, 0.2), 6 * wv(1, t, 4.1))
    rgb = R[..., :3]
    sp = (t - 3.6) / 1.4                                   # glint sweeps across the sign once per loop
    if 0 < sp < 1:
        pos = 900 + sp * 1300
        g = np.exp(-((diag - pos) ** 2) / (2 * 55 ** 2)) * (R[..., 4] / 255)
        rgb = np.clip(rgb + 120 * g[..., None], 0, 255)
    f = over(f, rgb, R[..., 3] / 255)
    # camera
    z = 1 + 0.03 * (0.5 - 0.5 * math.cos(TAU * t / P))
    M = np.array([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H / 2]], np.float32)
    out = cv2.warpAffine(np.clip(f, 0, 255).astype(np.uint8), M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if i in (0, 60, 120, 135, 150, 240, 299): Image.fromarray(out).save(f'_f{i:03d}.png')
    enc.stdin.write(out.tobytes())
enc.stdin.close(); enc.wait()
