"""Miller Daddy art: Miller gets a second hand (his own, lifted from the art and turned palm-down, on a jacket
sleeve reaching from his shoulder) that pats Trump on the head: three pats right away, three more mid-loop.
Trump's head dips a touch on each pat. Slow push-in. The hand rests on his head at the loop point, so the loop is
seamless (frame 0 = the art with the hand resting). Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image
from scipy import ndimage
import pat_proto as pp                     # hand cut-out + sleeve builder

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
a = pp.a
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
WX, WY = 1010.0, 430.0                     # wrist position at rest (fingers on his hair)
SHOULDER = (1340.0, 560.0)
# Trump's head region for the dip: soft blob around his head, fading to zero at the collar
head = np.zeros((H, W), np.float32); cv2.ellipse(head, (845, 560), (150, 120), 0, 0, 360, 1.0, -1)
head = ndimage.gaussian_filter(head, 25) * np.clip((700 - yy) / 120, 0, 1)

PATS = [(0.15, 3), (4.6, 3)]               # (start, number of pats), each pat 0.42s
def pat(t):
    for t0, n in PATS:
        u = (t - t0) / 0.42
        if 0 <= u < n:
            k = u - int(u); return math.sin(math.pi * k) ** 1.5   # 0 on the head, 1 lifted
    return 0.0
def contact(t):                            # head dip: brief, right as the hand lands
    for t0, n in PATS:
        u = (t - t0) / 0.42
        if 0 <= u < n + 0.15:
            k = u - int(u)
            if k < 0.18 and u >= 0.9: return math.sin(math.pi * k / 0.18)
    return 0.0

enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                       stdin=subprocess.PIPE)
for i in range(N):
    t = i / FPS
    lift = pat(t); dip = contact(t)
    f = a.copy()
    if dip > 0:                            # Trump's head pushed down a few px
        f = cv2.remap(f, xx, yy - 5 * dip * head, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    wx, wy = WX + 8 * lift, WY - 34 * lift + 4 * dip
    f = pp.over(f, pp.sleeve_layer(wx + 10, wy + 8, SHOULDER[0], SHOULDER[1], 64, 96))
    lay, _ = pp.hand_layer(78 - 9 * lift, 0.8, wx, wy)
    f = pp.over(f, lay)
    z = 1 + 0.035 * (0.5 - 0.5 * math.cos(TAU * t / P))
    Mc = np.float32([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H * 0.45]])
    out = cv2.warpAffine(np.clip(f, 0, 255).astype(np.uint8), Mc, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if i in (0, 10, 16, 299): Image.fromarray(out).save(f'_f{i:03d}.png')
    enc.stdin.write(out.tobytes())
enc.stdin.close(); enc.wait()
