"""Two Biggest Rats art: Trump keeps an eye on them. His eyes dart from the Vance rat to the Musk rat and back,
narrowing into a squint as he looks; the rats get nervous: heads flinch when he looks their way, ears flick,
bodies breathe fast, tails twitch. Everything is a smooth local warp of the art (no cut-outs, nothing uncovered,
so nothing can tear); slow push-in. Frame 0 = the art and every motion returns to rest by the loop point.
Run from art_build/. Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
art = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

def bump(cx, cy, rx, ry, p=2):
    r2 = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    return np.clip(1 - r2, 0, 1) ** p
EYES = [(705, 230, 135, 34), (1232, 216, 118, 40)]            # eye openings (centre, half-width, half-height)
def plateau(cx, cy, rx, ry, inner=0.6):
    r = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    u = np.clip((1 - r) / (1 - inner), 0, 1); return u * u * (3 - 2 * u)      # flat over the iris, easing to 0 at the corners
EYE_W = [plateau(*e) for e in EYES]
LID_W = [bump(cx, cy, rx * 1.15, ry * 2.4, 2) for cx, cy, rx, ry in EYES]
HEADS = [(672, 805, 105, 95), (1262, 800, 105, 95)]          # rat heads (Vance, Musk)
HEAD_W = [bump(*h) for h in HEADS]
EARS = [[(612, 748), (764, 752)], [(1200, 732), (1342, 742)]]
EAR_W = [[bump(x, y, 42, 46) for x, y in ee] for ee in EARS]
BODY = [(672, 900, 190, 150), (1262, 895, 200, 150)]
BODY_W = [bump(*b, p=1) for b in BODY]
TAIL_L = ((xx < 530) & (xx > 230)).astype(np.float32) * bump(380, 945, 200, 42, 1) * np.clip((530 - xx) / 120, 0, 1)
TAIL_R = ((xx > 1470) & (xx < 1880)).astype(np.float32) * bump(1680, 940, 260, 42, 1) * np.clip((xx - 1470) / 120, 0, 1)
ledge = (yy < 992).astype(np.float32)                         # never warp the white ledge edge

def ease(u): u = min(1, max(0, u)); return u * u * (3 - 2 * u)
# eye target over time: -1 = looking at Vance (left), +1 = at Musk (right), 0 = ahead
KEYS = [(0.0, 0), (0.2, 0), (0.32, -1), (1.05, -1), (1.17, 1), (2.0, 1), (2.15, 0), (4.9, 0), (5.02, 1), (5.7, 1),
        (5.82, -1), (6.6, -1), (6.75, 1), (7.3, 1), (7.45, 0), (10.0, 0)]
def look(t):
    for (t0, v0), (t1, v1) in zip(KEYS, KEYS[1:]):
        if t0 <= t <= t1: return v0 + (v1 - v0) * ease((t - t0) / max(t1 - t0, 1e-6))
    return 0.0
def squint(t):                                                # narrows while he's looking at one of them
    return abs(look(t)) * 0.16

def fields(t):
    dx = np.zeros((H, W), np.float32); dy = np.zeros((H, W), np.float32)
    lk = look(t); sq = squint(t)
    for k, (cx, cy, rx, ry) in enumerate(EYES):
        dx += lk * 17 * EYE_W[k]                              # iris slides toward the rat
        dy += -abs(lk) * 5 * EYE_W[k]                         # and down a touch (the rats are below)
        dy += sq * (yy - cy) * LID_W[k]                       # lids close in
    for k in range(2):
        towards = 1 if k == 0 else -1
        attn = max(0.0, -lk) if k == 0 else max(0.0, lk)      # Trump is looking at this rat
        # flinch: head pulls back/down and turns away when looked at; nervous jitter otherwise
        jit = 3.0 * math.sin(TAU * (2.3 + 0.7 * k) * t) * math.sin(TAU * 0.5 * t) ** 2
        dx += (towards * 9 * attn + jit) * HEAD_W[k]
        dy += -7 * attn * HEAD_W[k]
        # ears flick (quick, out of step)
        for j, w in enumerate(EAR_W[k]):
            f = math.sin(TAU * 0.5 * t + j * 1.7 + k) ** 24 * min(1, t / 0.4, (P - t) / 0.4)   # brief flicks
            dy += 9 * f * w; dx += (-1 if j == 0 else 1) * 4 * f * w
        # fast breathing
        br = 0.012 * (math.sin(TAU * 3 * t) - 0) * min(1, t / 0.3) * min(1, (P - t) / 0.3)
        bx, by = BODY[k][0], BODY[k][1]
        dx += -(xx - bx) * br * BODY_W[k]; dy += -(yy - by) * br * BODY_W[k]
    # tails twitch (travelling waves, zero at the loop point)
    w1 = TAU * 2 / P * t
    dy += 7 * (np.sin(xx / 45 - 3 * w1) - np.sin(xx / 45)) * TAIL_L
    dy += 7 * (np.sin(xx / 55 + 3 * w1) - np.sin(xx / 55)) * TAIL_R
    return dx * ledge, dy * ledge

def frame(t):
    dx, dy = fields(t)
    f = cv2.remap(art, xx - dx, yy - dy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    e = 0.5 - 0.5 * math.cos(TAU * t / P); z = 1 + 0.06 * e
    Mz = np.float32([[z, 0, (1 - z) * W * 0.5], [0, z, (1 - z) * H * 0.3]])
    return cv2.warpAffine(f, Mz, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        for t in map(float, sys.argv[1:]): Image.fromarray(np.clip(frame(t), 0, 255).astype(np.uint8)).save(f'_t{t:.2f}.png')
        sys.exit()
    enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'], stdin=subprocess.PIPE)
    for i in range(N): enc.stdin.write(np.clip(frame(i / FPS), 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close(); enc.wait()
