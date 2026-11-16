"""FAFO Era art: the red F-A-F-O letters punch in sequence, left to right (a quick red-hot flash each), right
away and again mid-loop; parallax push-in, Huang (rembg cut-out) pushing in faster than the face-and-letters
background. The letters are tinted shapes partly covered by the face, so the punch is light, not geometry.
Frame 0 = the art. Run from art_build/. Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image
from rembg import remove, new_session

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
im = Image.open('still_1920.png').convert('RGB'); art = np.array(im).astype(np.float32)
FG = np.array(remove(im, session=new_session('isnet-general-use'), only_mask=True)).astype(np.float32) / 255
_b = (FG > 0.5).astype(np.uint8)                     # Huang only: drop the matte's faint stray values elsewhere
_n, _lab, _st, _ = cv2.connectedComponentsWithStats(_b, 8)
_b = (_lab == 1 + int(np.argmax(_st[1:, 4]))).astype(np.float32)
FG = np.where(cv2.dilate(_b, np.ones((5, 5), np.uint8)) > 0, FG, 0) * (FG > 0.15)
xx = np.arange(W, dtype=np.float32)[None, :]

# red letters (soft mask from how red a pixel is), not under Huang
r, g, b = art[..., 0], art[..., 1], art[..., 2]
RED = np.clip((r - np.maximum(g, b) - 40) / 50, 0, 1) * (1 - FG)
LETTERS = [300, 640, 1330, 1720]                    # rough centres of F, A, F, O (px)
BEATS = [0.15, 5.0]                                  # each run of four punches
def punch(t):
    k = np.zeros((1, W), np.float32)
    for t0 in BEATS:
        for i, cx in enumerate(LETTERS):
            s = (t - t0 - i * 0.14) / 0.32
            if 0 <= s < 1:
                amp = (1 - s) ** 2 * min(1, s * 8)   # snaps on, decays
                k = np.maximum(k, amp * np.exp(-((xx - cx) / 190) ** 2))
    return k

# background behind Huang, for the parallax (inpainted at half size; only a thin band is ever seen)
hole = cv2.dilate((FG > 0.02).astype(np.uint8), np.ones((7, 7), np.uint8)) * 255
small = cv2.resize(art.astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_AREA)
plate_s = cv2.inpaint(small, cv2.resize(hole, (W // 2, H // 2)), 6, cv2.INPAINT_TELEA)
RED_P = cv2.resize(cv2.resize(RED, (W // 2, H // 2)), (W, H))   # letters carry on behind him
plate = np.where((hole > 0)[..., None], cv2.resize(plate_s, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32), art)
RED_ALL = np.where(hole > 0, np.clip((plate[..., 0] - np.maximum(plate[..., 1], plate[..., 2]) - 40) / 50, 0, 1), RED)

CX, CY = 0.5 * W, 0.45 * H
def cam(t, depth):
    e = 0.5 - 0.5 * math.cos(TAU * t / P)
    z = 1 + 0.07 * e * depth
    return np.float32([[z, 0, (1 - z) * CX], [0, z, (1 - z) * CY]])

def frame(t):
    k = punch(t)
    bg = plate
    if k.max() > 0.001:
        kk = (k * RED_ALL)[..., None]
        hot = np.stack([np.minimum(255, plate[..., 0] * 1.4 + 70), plate[..., 1] * 1.8 + 95, plate[..., 2] * 1.7 + 80], 2)   # red-hot: toward a hot pink-white
        bg = plate * (1 - kk) + hot * kk
    bgw = cv2.warpAffine(bg, cam(t, 0.35), (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    Mf = cam(t, 1.0)
    fa = cv2.warpAffine(FG, Mf, (W, H), flags=cv2.INTER_LINEAR)[..., None]
    fc = cv2.warpAffine(art * FG[..., None], Mf, (W, H), flags=cv2.INTER_CUBIC)
    return bgw * (1 - fa) + fc

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        for t in map(float, sys.argv[1:]): Image.fromarray(np.clip(frame(t), 0, 255).astype(np.uint8)).save(f'_t{t:.2f}.png')
        sys.exit()
    enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                           stdin=subprocess.PIPE)
    for i in range(N): enc.stdin.write(np.clip(frame(i / FPS), 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close(); enc.wait()
