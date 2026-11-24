"""Fetterman art: the EXIT sign behind him buzzes like a failing tube: a stutter of flickers right away (0.15-0.9s),
then it hums with a faint unsteady glow, and stutters again at 5s and 7.6s. When it dims, the letters lose their
glow first and the red panel darkens; the red light it spills on the backdrop dims with it. Fetterman (rembg) and
the Capitol are untouched; a slow push-in on him throughout. Frame 0 = the art. Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image
from rembg import remove, new_session

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
im = Image.open('still_1920.png').convert('RGB'); art = np.array(im).astype(np.float32)
PER = np.array(remove(im, session=new_session('u2net_human_seg'), only_mask=True)).astype(np.float32) / 255
r, g, b = art[..., 0], art[..., 1], art[..., 2]
sat = art.max(2) - art.min(2)
red = np.clip((r - np.maximum(g, b) - 40) / 60, 0, 1)                       # the sign's red, and its spill
letters = np.clip((g - 150) / 50, 0, 1) * np.clip((r - b - 40) / 40, 0, 1)  # warm cream letters
xx = np.arange(W)[None, :].repeat(H, 0)
SIGN = np.clip(np.maximum(red, letters), 0, 1) * (xx > 700) * (1 - PER)
SIGN = cv2.GaussianBlur(SIGN.astype(np.float32), (0, 0), 1.5)
LET = cv2.GaussianBlur(letters * (xx > 700) * (1 - PER), (0, 0), 1.5)[..., None]
SIGN = SIGN[..., None]

STUTTERS = [(0.15, [0.0, 0.06, 0.13, 0.22, 0.27, 0.42, 0.5, 0.62]), (5.0, [0.0, 0.05, 0.14, 0.2]), (7.6, [0.0, 0.07, 0.12, 0.3])]
def level(t):
    """1 = fully lit, lower = dim"""
    for t0, cuts in STUTTERS:
        u = t - t0
        if 0 <= u < cuts[-1] + 0.12:
            k = sum(1 for c in cuts if c <= u)                               # toggles off/on at each cut
            if k % 2 == 1: return 0.25 + 0.1 * math.sin(u * 80)
    hum = 0.03 * math.sin(TAU * 7 * t) * math.sin(TAU * 0.37 * t) if t > 1.0 else 0.0
    return 1.0 + hum

def frame(t):
    lv = level(t)
    f = art.copy()
    if lv != 1.0:
        d = (1 - lv)
        dim = art * (1 - 0.75 * d) - LET * 90 * d                           # letters lose their glow first
        f = art * (1 - SIGN) + np.clip(dim, 0, 255) * SIGN
    e = 0.5 - 0.5 * math.cos(TAU * t / P); z = 1 + 0.045 * e
    M = np.float32([[z, 0, (1 - z) * W * 0.44], [0, z, (1 - z) * H * 0.4]])
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
