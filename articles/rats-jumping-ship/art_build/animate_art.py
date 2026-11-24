"""Two Biggest Rats art: the art holds still (Trump's glare, both rats untouched) while two dark silhouette hands
slam down from the top of the frame and pin the rats by their tails: the left hand on Vance's tail at 0.3s, the
right on Musk's at 0.55s, each with its shadow sharpening as it drops and a jolt of camera shake on impact. They
press and tremble, lift and slam again at 5s, and lift clear out of frame before the loop point.
Slow push-in throughout. Frame 0 = the art. Run from art_build/. Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
art = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)

def hand_mask(mirror=False):
    """Palm-down hand seen from above, fingertips pointing down; forearm runs off the top. 4x supersampled.
    Returns (alpha HxW float, fingertip point (x, y) in the mask's own coords)."""
    S = 4; w, h = 360, 1500
    m = np.zeros((h * S, w * S), np.uint8)
    cx = w / 2
    def P(x, y): return (int(x * S), int(y * S))
    # forearm and palm
    cv2.fillPoly(m, [np.int32([P(cx - 72, 0), P(cx + 72, 0), P(cx + 92, 1020), P(cx - 92, 1020)])], 255, cv2.LINE_AA)
    cv2.ellipse(m, P(cx, 1110), (int(108 * S), int(130 * S)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    # fingers (index..pinky): x offset, length, width
    fingers = [(-66, 170, 40), (-22, 205, 42), (22, 195, 41), (64, 150, 36)]
    for dx, ln, fw in fingers:
        x0, y0 = cx + dx * 0.95, 1150; x1, y1 = cx + dx * 1.12, 1150 + ln
        cv2.line(m, P(x0, y0), P(x1, y1), 255, int(fw * S), cv2.LINE_AA)
        cv2.circle(m, P(x1, y1), int(fw * S / 2), 255, -1, cv2.LINE_AA)
    # thumb, out to the side
    cv2.line(m, P(cx - 95, 1060), P(cx - 160, 1190), 255, int(44 * S), cv2.LINE_AA)
    cv2.circle(m, P(cx - 160, 1190), int(22 * S), 255, -1, cv2.LINE_AA)
    a = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    if mirror: a = a[:, ::-1]
    tip = (cx, 1150 + 205)
    return a, tip

HANDS = []
for (tx, ty, ang, mirror) in ((330, 955, 14, False), (1700, 948, -14, True)):   # where each hand's middle fingertip lands
    a, tip = hand_mask(mirror)
    HANDS.append(dict(a=a, tip=tip, target=(tx, ty), ang=ang))
INK = np.array([16, 14, 20], np.float32)

def drop(k, t):
    """0 = landed, larger = raised (px up), for hand k"""
    t_in = [0.3, 0.55][k]
    up = 1100.0
    for s0, s1 in ((t_in, 4.6 + 0.2 * k), (5.0 + 0.25 * k, 8.7 + 0.15 * k)):
        if t < s0 - 0.2: continue
        if t < s0: u = (t - (s0 - 0.2)) / 0.2; return up * (1 - u * u)          # accelerating down
        if t < s1: return -3.0 * math.sin(TAU * 9 * (t - s0)) * math.exp(-(t - s0) * 6) + 1.2 * math.sin(TAU * 3.3 * t)  # pressing, trembling
        if t < s1 + 0.35: u = (t - s1) / 0.35; return up * (u * u)               # lifted away
    return up

def place(k, rise):
    h = HANDS[k]; a = h['a']; tx, ty = h['target']
    M = cv2.getRotationMatrix2D(h['tip'], h['ang'], 1.0)
    M[0, 2] += tx - h['tip'][0]; M[1, 2] += ty - h['tip'][1] - rise
    return cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR)

def shake(t):
    s = 0.0
    for t0 in (0.3, 0.55, 5.0, 5.25):
        u = t - t0
        if 0 <= u < 0.3: s += 7 * math.exp(-u * 14) * math.sin(TAU * 22 * u)
    return s

def frame(t):
    f = art.copy()
    for k in range(2):
        rise = drop(k, t)
        if rise > 1090: continue
        al = place(k, rise)
        # shadow on the scene: offset and soft while high, tight when landed
        hgt = min(1.0, max(0.0, rise) / 400)
        sh = place(k, 0) if rise < 1 else place(k, rise * 0.25)
        sh = cv2.GaussianBlur(sh, (0, 0), 6 + 30 * hgt) * (0.55 - 0.3 * hgt)
        sh = np.roll(np.roll(sh, int(14 + 40 * hgt), 0), int(10 + 30 * hgt), 1)
        f = f * (1 - sh[..., None])
        f = f * (1 - 0.94 * al[..., None]) + INK * 0.94 * al[..., None]
        # a rim of light around the silhouette, so it reads against the dark suit
        rim = np.clip(al - cv2.erode(al, np.ones((7, 7), np.uint8)), 0, 1)
        rim = cv2.GaussianBlur(rim, (0, 0), 1.2)[..., None]
        f = f * (1 - 0.75 * rim) + np.array([235, 215, 190], np.float32) * 0.75 * rim
        glow = cv2.GaussianBlur(al, (0, 0), 14)[..., None] * (1 - al[..., None])
        f = 255 - (255 - f) * (1 - 0.18 * glow)
    sx = shake(t)
    e = 0.5 - 0.5 * math.cos(TAU * t / P); z = 1 + 0.05 * e
    Mz = np.float32([[z, 0, (1 - z) * W * 0.5 + sx], [0, z, (1 - z) * H * 0.35 + sx * 0.6]])
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
