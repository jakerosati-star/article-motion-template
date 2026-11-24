"""Trump / American Communism art: a slow push-in, and Trump's oversized head bobbles like a bobblehead: knocked at
0.15s it wobbles side to side on the neck and settles (a damped spring), knocked again at 5s. The head (rembg matte,
skin and hair above the collar) rotates about the neck; the slivers it uncovers are inpainted. Frame 0 = the art.
Run from art_build/. Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
art = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
TR = np.load('rb.npy').astype(np.float32) / 255
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
lum = art.mean(2); r, g, b = art[..., 0], art[..., 1], art[..., 2]
# head: Trump's skin and hair above the collar (the suit is dark, the collar white; the head is warm/bright)
box = (yy > 250) & (yy < 518) & (xx > 900) & (xx < 1120)
head = ((TR > 0.5) & box & (lum > 80) & ~((r > 220) & (g > 220) & (b > 220) & (yy > 470))).astype(np.uint8)
head = cv2.morphologyEx(head, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(head, 8)
head = (lab == 1 + int(np.argmax(st[1:, 4]))).astype(np.uint8)
from scipy import ndimage
head = ndimage.binary_fill_holes(head).astype(np.float32)
HEAD = cv2.GaussianBlur(head, (0, 0), 1.0) * TR
ys, xs = np.where(HEAD > 0.5)
PIV = (float(xs.mean()), float(ys.max()) - 6)                    # the neck, just above the collar
hole = cv2.dilate((HEAD > 0.02).astype(np.uint8), np.ones((9, 9), np.uint8))
plate = cv2.inpaint(art.astype(np.uint8), hole * 255, 9, cv2.INPAINT_TELEA).astype(np.float32)
# below the chin the body continues: keep the art there (the head only uncovers background at its sides/top)
plate = np.where((hole > 0)[..., None], plate, art)
HC = art * HEAD[..., None]

def angle(t):
    a = 0.0
    for t0, amp in ((0.15, 9.0), (5.0, 7.0)):
        u = t - t0
        if u >= 0: a += amp * math.exp(-u * 1.6) * math.sin(TAU * 1.9 * u)
    return a * (1 - min(1, max(0, (t - 9.0) / 1.0)))              # fully settled by the loop point

# the rebuilt background gets the flag's speckle back: its fine detail, borrowed from the area just around the head
ring = (cv2.dilate(hole, np.ones((61, 61), np.uint8)) > 0) & (hole == 0) & (TR < 0.1)
hp = art - cv2.GaussianBlur(art, (0, 0), 3)
ys_, xs_ = np.where(ring)
idx = (yy.astype(np.int64) * 7919 + xx.astype(np.int64) * 104729) % len(ys_)
tex = hp[ys_[idx], xs_[idx]]
plate = np.where((hole > 0)[..., None], plate + tex * 0.8, plate)
HARD = (HEAD > 0.5).astype(np.float32)
def frame(t):
    a = angle(t)
    if abs(a) < 0.05: f = art.copy()
    else:
        M = cv2.getRotationMatrix2D(PIV, a, 1.0)
        f = np.where((hole > 0)[..., None], plate, art)
        ha = cv2.warpAffine(HEAD, M, (W, H))[..., None]; hc = cv2.warpAffine(HC, M, (W, H), flags=cv2.INTER_CUBIC)
        f = f * (1 - ha) + hc
        # where the rotated head doesn't reach, below the chin, the body (collar, suit) is the art's own
        keep = ((yy > PIV[1] - 2) & (HEAD < 0.5) & (TR > 0.5))[..., None]
        f = np.where(keep, art, f)
    e = 0.5 - 0.5 * math.cos(TAU * t / P); z = 1 + 0.06 * e
    Mz = np.float32([[z, 0, (1 - z) * PIV[0]], [0, z, (1 - z) * PIV[1]]])
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
