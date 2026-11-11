"""Medicare Fight art: the two speakers lean in toward each other in the first second (squaring off), ease apart,
square off again mid-loop; the pointing arm jabs, the yellow circle pulses, slow push-in. Concrete + circle rebuilt
behind them. Seamless 10s loop; frame 0 is the original. Run layers.py first. Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
bg = np.array(Image.open('bg.png').convert('RGB')).astype(np.float32)
PC = dict(np.load('pieces.npz'))
LEFT = np.dstack([a, PC['left'] * 255]); RIGHT = np.dstack([a, PC['right'] * 255])
cx, cy, r = 960, 540, 494
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
circ = np.clip((r - np.hypot(xx - cx, yy - cy)) / 3, 0, 1) * (1 - np.maximum(PC['left'], PC['right']))

def ss(x): x = max(0.0, min(1.0, x)); return x * x * (3 - 2 * x)
def wv(k, t, ph=0): return math.sin(TAU * k * t / P + ph)
def face_off(t):
    """0 -> 1 (squared off) -> 0, twice per loop"""
    a1 = ss(t / 0.7) * (1 - ss((t - 2.8) / 1.0))
    a2 = ss((t - 5.0) / 0.6) * (1 - ss((t - 8.2) / 1.2))
    return max(a1, a2)
def jab(t):   # the pointing arm thrusts up twice per face-off
    s = 0.0
    for t0 in (0.5, 1.4, 5.5, 6.3):
        u = (t - t0) / 0.35
        if 0 <= u <= 1: s = max(s, math.sin(math.pi * u))
    return s

def put(f, layer, M):
    w = cv2.warpAffine(layer, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0); al = w[..., 3] / 255
    return f * (1 - al[..., None]) + w[..., :3] * al[..., None]

enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                       stdin=subprocess.PIPE)
for i in range(N):
    t = i / FPS
    if i == 0: enc.stdin.write(a.astype(np.uint8).tobytes()); continue
    k = face_off(t); j = jab(t)
    f = bg.copy()
    pul = 0.5 - 0.5 * math.cos(TAU * 4 * t / P)                      # circle pulse: brightens and glows
    f = f * (1 + 0.10 * pul * circ)[..., None]
    # right speaker leans in toward the podium (pivot at his feet), behind the podium group
    Mr = cv2.getRotationMatrix2D((1300.0, 940.0), 4.0 * k, 1.0 + 0.03 * k); Mr[0, 2] -= 45 * k
    f = put(f, RIGHT, Mr)
    # podium group leans toward him; the pointing arm adds a jab (whole group: the arm is part of the same cut-out)
    Ml = cv2.getRotationMatrix2D((700.0, 1000.0), -2.5 * k - 1.2 * j, 1.0 + 0.02 * k); Ml[0, 2] += 30 * k; Ml[1, 2] -= 6 * j
    f = put(f, LEFT, Ml)
    z = 1 + 0.03 * (0.5 - 0.5 * math.cos(TAU * t / P))
    Mc = np.float32([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H / 2]])
    out = cv2.warpAffine(np.clip(f, 0, 255).astype(np.uint8), Mc, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if i in (21, 45, 120, 170, 299): Image.fromarray(out).save(f'_f{i:03d}.png')
    enc.stdin.write(out.tobytes())
enc.stdin.close(); enc.wait()
