"""Right's Meltdown art: flames lick upward and flicker around a still microphone, embers rise, slow push-in.
Seamless 10s loop; frame 0 is the original. Writes ../art_loop.mp4"""
import math, random, subprocess
import numpy as np, cv2
from PIL import Image
from scipy import ndimage

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
m = np.array(Image.open('rembg_mask.png')).astype(np.float32) / 255
sat = a.max(2) - a.min(2)
mic = ndimage.binary_fill_holes(ndimage.binary_closing((m > 0.5) & (sat < 70), iterations=3))
lab, n = ndimage.label(mic); mic = lab == 1 + int(np.argmax(ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))))
mic_a = ndimage.gaussian_filter(mic.astype(np.float32), 1.0)
fire = ndimage.gaussian_filter(((m > 0.3) & ~mic).astype(np.float32), 25)     # where the flames are, softened
fire = np.clip(fire * 1.6, 0, 1).astype(np.float32)
hot = np.clip((sat - 120) / 100, 0, 1) * (m > 0.3)                             # saturated flame pixels, for flicker
hot = ndimage.gaussian_filter(hot.astype(np.float32), 3)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

rng = np.random.default_rng(4)
WAVES = [(rng.uniform(0.010, 0.022), rng.uniform(0.004, 0.012), rng.uniform(0, TAU), int(rng.integers(2, 5))) for _ in range(5)]
def flow(t):
    """upward-travelling ripples: flames lick up and sway (periodic over P)"""
    dx = np.zeros((H, W), np.float32); dy = np.zeros((H, W), np.float32)
    for ky, kx, ph, nn in WAVES:
        arg = yy * ky + xx * kx + ph + TAU * nn * t / P
        dx += 9 * np.sin(arg); dy += 7 * np.cos(arg * 0.8 + 1.1)
    dx /= len(WAVES) / 2; dy /= len(WAVES) / 2
    return dx * fire, dy * fire

er = random.Random(7); Ly = 900
embers = [dict(x=er.gauss(960, 330), y0=er.uniform(0, Ly), n=er.choice([1, 2, 2, 3]), sw=er.uniform(8, 26), m=er.randint(1, 3),
               ph=er.uniform(0, TAU), r=er.choice([2, 2, 2.5, 3, 4]), k=er.randint(3, 8)) for _ in range(70)]
def wv(k, t, ph=0): return math.sin(TAU * k * t / P + ph)
def over(base, rgb, al): return base * (1 - al[..., None]) + rgb * al[..., None]

enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                       stdin=subprocess.PIPE)
dx0, dy0 = flow(0.0)
for i in range(N):
    t = i / FPS
    dx, dy = flow(t); dx -= dx0; dy -= dy0                      # zero displacement at t = 0: frame 0 is the original
    f = cv2.remap(a, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fl = 1 + (0.07 * wv(11, t) * wv(4, t, 1.3) + 0.04 * wv(17, t, 0.4)) * hot
    f = np.clip(f * fl[..., None], 0, 255)
    el = np.zeros((H, W), np.float32)
    for e in embers:
        y = 760 - ((e['y0'] + Ly * e['n'] * t / P) % Ly)
        x = e['x'] + e['sw'] * wv(e['m'], t, e['ph'])
        al = (0.55 + 0.45 * wv(e['k'], t, e['ph'])) * min(1, (y + 140) / 250)
        if al > 0: cv2.circle(el, (int(x * 4), int(y * 4)), int(e['r'] * 4), al, -1, lineType=cv2.LINE_AA, shift=2)
    glow = cv2.GaussianBlur(el, (0, 0), 3) * 0.8 + el
    f = np.clip(f + glow[..., None] * np.array([255, 190, 70], np.float32), 0, 255)
    f = over(f, a, mic_a)                                        # the mic stays put, on top
    z = 1 + 0.035 * (0.5 - 0.5 * math.cos(TAU * t / P))
    Mc = np.float32([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H * 0.45]])
    out = cv2.warpAffine(f.astype(np.uint8), Mc, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if i in (0, 75, 150, 225, 299): Image.fromarray(out).save(f'_f{i:03d}.png')
    enc.stdin.write(out.tobytes())
enc.stdin.close(); enc.wait()
