"""GOP Coping art: the frowning balloon bobs from the first frame, then slowly deflates, sags and droops on its
string, and re-inflates before the loop point. Trump stays still; slow push-in. Seamless 10s loop; frame 0 is the
original. Run layers.py first. Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
bg = np.array(Image.open('bg.png').convert('RGB')).astype(np.float32)
PC = dict(np.load('pieces.npz'))
BAL = np.dstack([a, PC['balloon'] * 255]).astype(np.float32)
STR = np.dstack([a, PC['string'] * 255]).astype(np.float32)
KNOT = (504.0, 593.0)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

def ss(x): x = max(0.0, min(1.0, x)); return x * x * (3 - 2 * x)
def back_out(x, s=1.6): x = max(0.0, min(1.0, x)) - 1; return 1 + (s + 1) * x ** 3 + s * x ** 2
def wv(k, t, ph=0): return math.sin(TAU * k * t / P + ph)
def deflate(t):   # 0 = full, 1 = sagged. Starts right away, slow leak, quick re-inflate with a bounce
    d = ss((t - 0.3) / 6.4)
    return d * (1 - back_out((t - 7.6) / 1.6))

def place(f, layer, M):
    w = cv2.warpAffine(layer, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0); al = w[..., 3] / 255
    return f * (1 - al[..., None]) + w[..., :3] * al[..., None]

enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                       stdin=subprocess.PIPE)
for i in range(N):
    t = i / FPS
    if i == 0: enc.stdin.write(a.astype(np.uint8).tobytes()); continue
    d = deflate(t)
    bob = 14 * math.sin(TAU * 3 * t / P) * (1 - 0.6 * d)        # lively bob while full, sluggish when sagging
    sway = 5 * math.sin(TAU * 2 * t / P + 0.4) * (1 - 0.5 * d) - 5 * math.sin(0.4)
    drop = 120 * d                                                # sinks as it loses air
    tilt = sway + 14 * d                                          # droops over to one side
    kx, ky = KNOT[0] + 10 * d, KNOT[1] + drop + bob
    f = bg.copy()
    # string: top follows the knot, bottom end stays roughly put
    wgt = np.clip(1 - (yy - KNOT[1]) / 420, 0, 1)
    dxs, dys = (kx - KNOT[0]) * wgt + 6 * math.sin(TAU * 2 * t / P) * wgt * (1 - wgt) * 4, (ky - KNOT[1]) * wgt
    sw = cv2.remap(STR, xx - dxs.astype(np.float32), yy - dys.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
    al = sw[..., 3] / 255; f = f * (1 - al[..., None]) + sw[..., :3] * al[..., None]
    # balloon: shrinks toward the knot, squashes a little, darkens as it goes soft
    s = 1 - 0.34 * d; sx, sy = s * (1 + 0.10 * d), s * (1 - 0.12 * d)
    th = math.radians(tilt); c, sn = math.cos(th), math.sin(th)
    A = np.array([[c * sx, -sn * sy], [sn * sx, c * sy]], np.float32)
    M = np.hstack([A, (np.array([kx, ky]) - A @ np.array(KNOT)).reshape(2, 1)]).astype(np.float32)
    lay = BAL.copy(); lay[..., :3] *= 1 - 0.18 * d
    f = place(f, lay, M)
    z = 1 + 0.03 * (0.5 - 0.5 * math.cos(TAU * t / P))
    Mc = np.float32([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H / 2]])
    out = cv2.warpAffine(np.clip(f, 0, 255).astype(np.uint8), Mc, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if i in (15, 120, 210, 255, 285): Image.fromarray(out).save(f'_f{i:03d}.png')
    enc.stdin.write(out.tobytes())
enc.stdin.close(); enc.wait()
