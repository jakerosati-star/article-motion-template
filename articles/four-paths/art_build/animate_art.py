"""Four Paths art: the elephant tugs the rope twice per loop, the gold statue tips toward it and wobbles back.
Seamless 10s loop, full image from frame 0 (rest pose = the original). Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image
from scipy import ndimage

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
m = np.array(Image.open('rembg_mask.png')).astype(np.float32) / 255
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

# drop the patch of dark tree rembg grabbed behind the trunk
lum = a.mean(2)
m[(xx > 300) & (xx < 480) & (yy > 530) & (yy < 700) & (lum < 75)] = 0

T0 = np.array([300.0, 556.0])     # rope meets the trunk
S0 = np.array([1180.0, 326.0])    # rope meets the statue
d = S0 - T0; L = np.linalg.norm(d); u = d / L; nrm = np.array([-u[1], u[0]])
along = (xx - T0[0]) * u[0] + (yy - T0[1]) * u[1]; across = (xx - T0[0]) * nrm[0] + (yy - T0[1]) * nrm[1]
rope_band = (np.abs(across) < 11) & (along > 22) & (along < L + 6)

PED_TOP = 762
rope_a = m * rope_band
fig_a = m * (~rope_band) * (xx > 1140) * (yy < PED_TOP)
ped_a = m * (xx > 1140) * (yy >= PED_TOP)
ele_a = m * (~rope_band) * (xx < 900) * (yy > 500)
lab, n = ndimage.label(ele_a > 0.3)                      # elephant: keep the body, drop stray tree specks
if n > 1:
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)); keep = lab == (1 + int(np.argmax(sizes)))
    ele_a[~ndimage.binary_dilation(keep, iterations=2)] = 0
for arr in (fig_a, ele_a):
    arr[:] = np.where(ndimage.binary_erosion(arr > 0.5, iterations=2), 1.0, arr)

# background: fill behind everything that moves (rope, statue figure, elephant)
hole = cv2.dilate(((rope_a + fig_a + ele_a) > 0.05).astype(np.uint8), np.ones((13, 13), np.uint8))
bg = cv2.inpaint(a.astype(np.uint8), hole, 12, cv2.INPAINT_TELEA).astype(np.float32)
bg = over_ = None or bg
def over(base, rgb, al): return base * (1 - al[..., None]) + rgb * al[..., None]
bg = over(bg, a, ped_a)            # pedestal stays put

FIG = np.dstack([a, fig_a * 255]); ELE = np.dstack([a, ele_a * 255]); ROPE = np.dstack([a, rope_a * 255])
FIG_PIVOT = (1340.0, 762.0); ELE_PIVOT = (840.0, 1000.0)

def clamp(x): return max(0.0, min(1.0, x))
def ss(x): x = clamp(x); return x * x * (3 - 2 * x)
def tug(t, t0, pull, hold, rel, amp):
    """0 -> amp over `pull`s, hold, then release with a damped wobble back to 0."""
    if t < t0: return 0.0
    t -= t0
    if t < pull: return amp * ss(t / pull)
    t -= pull
    if t < hold: return amp
    t -= hold
    return amp * math.exp(-t / (rel * 0.45)) * math.cos(TAU * t / (rel * 0.9)) if t < rel * 3 else 0.0
def strain(t): return tug(t, 1.2, 0.9, 0.6, 0.9, 0.55) + tug(t, 5.0, 1.0, 0.9, 1.0, 1.0)

def aff(pivot, ang, dx, dy):
    M = cv2.getRotationMatrix2D(pivot, ang, 1.0); M[0, 2] += dx; M[1, 2] += dy; return M
def apply(M, p): return M[:, :2] @ p + M[:, 2]

enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                       stdin=subprocess.PIPE)
for i in range(N):
    t = i / FPS; s = strain(t)
    Me = aff(ELE_PIVOT, -1.6 * s, -9 * s, 0)          # elephant leans back, head up
    Mf = aff(FIG_PIVOT, 2.6 * s, 0, 0)                # statue tips toward the elephant
    T1, S1 = apply(Me, T0), apply(Mf, S0)
    # rope: similarity transform taking (T0, S0) to (T1, S1), so it stays taut between trunk and statue
    v0, v1 = S0 - T0, S1 - T1
    sc = np.linalg.norm(v1) / np.linalg.norm(v0); ang = math.degrees(math.atan2(v1[1], v1[0]) - math.atan2(v0[1], v0[0]))
    Mr = cv2.getRotationMatrix2D(tuple(T0), -ang, sc); Mr[:, 2] += T1 - T0
    f = bg.copy()
    for img, M in ((ROPE, Mr), (FIG, Mf), (ELE, Me)):
        w = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        f = over(f, w[..., :3], w[..., 3] / 255)
    z = 1 + 0.03 * (0.5 - 0.5 * math.cos(TAU * t / P))
    Mc = np.array([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H * 0.55]], np.float32)
    out = cv2.warpAffine(np.clip(f, 0, 255).astype(np.uint8), Mc, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if i in (0, 60, 195, 225, 299): Image.fromarray(out).save(f'_f{i:03d}.png')
    enc.stdin.write(out.tobytes())
enc.stdin.close(); enc.wait()
