"""Budget Nerds art: Vought shrinks down ("wings clipped") within the first second, the Capitol leans in over him,
the bill stack fans back and forth, the green panel pulses, slow push-in. Seamless 10s loop; frame 0 is the
original. Run layers.py, then fill_bills.py. Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
bg = np.array(Image.open('bg.png').convert('RGB')).astype(np.float32)
PC = dict(np.load('pieces.npz'))
L = {k: np.dstack([a, PC[k] * 255]).astype(np.float32) for k in ('vought', 'dome')}
# bills: the original stack with the part Vought hid rebuilt from a reference $100 note (fill_bills.py)
L['bills'] = np.dstack([np.load('bills_rgb.npy').astype(np.float32), PC['bills_full'] * 255])
green = np.zeros((H, W), np.float32); green[207:872, 308:1610] = 1
green *= 1 - np.maximum.reduce([PC['bills_full'], PC['dome'], PC['vought']])

def ss(x): x = max(0.0, min(1.0, x)); return x * x * (3 - 2 * x)
def wv(k, t, ph=0): return math.sin(TAU * k * t / P + ph)
def clip(t):   # 0 -> 1 -> 0: shrink by 1.0s, hold, recover 8.0-9.6s
    return ss(t / 0.8) * (1 - ss((t - 8.0) / 1.6))
def place(f, layer, M):
    w = cv2.warpAffine(layer, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0); al = w[..., 3] / 255
    return f * (1 - al[..., None]) + w[..., :3] * al[..., None]

enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                       stdin=subprocess.PIPE)
for i in range(N):
    t = i / FPS
    if i == 0: enc.stdin.write(a.astype(np.uint8).tobytes()); continue
    f = bg.copy()
    f = f + (18 * (0.5 - 0.5 * math.cos(TAU * 3 * t / P)) * green)[..., None] * np.array([0.3, 1, 0.4], np.float32)   # panel pulse
    c = clip(t)
    # bills fan: rock about the bottom-left corner of the stack
    ang = 4.0 * wv(2, t) + 1.5 * wv(4, t, 0.7) - 1.5 * math.sin(0.7)
    f = place(f, L['bills'], cv2.getRotationMatrix2D((470, 1060), ang, 1.0))
    # Capitol leans in toward him while he shrinks, plus a slow sway
    M = cv2.getRotationMatrix2D((1400, 950), 3.5 * c + 0.6 * wv(1, t, 1.0) - 0.6 * math.sin(1.0), 1.0); M[0, 2] -= 25 * c
    f = place(f, L['dome'], M)
    # Vought: shrinks from the feet (wings clipped), small sway
    s = 1 - 0.16 * c; sway = 1.2 * wv(2, t, 2.0) - 1.2 * math.sin(2.0)
    M = cv2.getRotationMatrix2D((1000, 1030), sway, s)
    f = place(f, L['vought'], M)
    z = 1 + 0.03 * (0.5 - 0.5 * math.cos(TAU * t / P))
    Mc = np.float32([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H / 2]])
    out = cv2.warpAffine(np.clip(f, 0, 255).astype(np.uint8), Mc, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if i in (15, 30, 120, 270): Image.fromarray(out).save(f'_f{i:03d}.png')
    enc.stdin.write(out.tobytes())
enc.stdin.close(); enc.wait()
