"""FAFO Era art. The art is a red block with the letters F-A-F-O knocked out of it (the face shows through the letter
shapes; the red is the space around and inside them), with the face also showing through the red as a tint.
Animation: the loop opens on the face alone (no red), then the red stamps in one letter-cell at a time, F, A, F, O,
in the first second, each with a small overshoot scale and a bright flash; the letters leave in reverse just before
the loop point. Parallax push-in: Huang (cut out) pushes in faster than the background.
Run from art_build/. Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image
from rembg import remove, new_session

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
im = Image.open('still_1920.png').convert('RGB'); art = np.array(im).astype(np.float32)
xx = np.arange(W, dtype=np.float32)[None, :]

# --- Huang: rembg matte, keeping his body and his separate left hand (pieces > 5000 px), dropping stray values
FG = np.array(remove(im, session=new_session('isnet-general-use'), only_mask=True)).astype(np.float32) / 255
_n, _lab, _st, _ = cv2.connectedComponentsWithStats((FG > 0.5).astype(np.uint8), 8)
_keep = np.isin(_lab, [i for i in range(1, _n) if _st[i, 4] > 5000]).astype(np.float32)
FG = np.where(cv2.dilate(_keep, np.ones((5, 5), np.uint8)) > 0, FG, 0) * (FG > 0.15)
FG = cv2.GaussianBlur((FG > 0.4).astype(np.float32), (0, 0), 0.8)    # a firm edge: half-transparent jacket pixels would tint
_redart = np.clip((art[..., 0] - np.maximum(art[..., 1], art[..., 2]) - 110) / 30, 0, 1) * (np.maximum(art[..., 1], art[..., 2]) < 40)   # the letters' pure red only, not skin
FG = FG * (1 - cv2.GaussianBlur(_redart, (0, 0), 0.8))                 # red seen through gaps (by his elbow) is background
# where he borders the red, a hard edge: a half-transparent pixel there would carry red into him (or tint when it goes)
_nr = cv2.dilate((np.clip((art[..., 0] - np.maximum(art[..., 1], art[..., 2]) - 30) / 30, 0, 1) > 0.3).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
_skin_or_jacket = FG > 0.5
FG = np.where(_nr, (_skin_or_jacket & (_redart < 0.3) & ~((art[..., 0] - np.maximum(art[..., 1], art[..., 2])) > 70) | (FG > 0.97)).astype(np.float32), FG)

# --- background behind Huang (inpainted; only thin bands are ever seen during the parallax)
hole = cv2.dilate((FG > 0.02).astype(np.uint8), np.ones((7, 7), np.uint8)) * 255
plate = cv2.inpaint(art.astype(np.uint8), hole, 6, cv2.INPAINT_TELEA).astype(np.float32)
near = cv2.dilate((FG > 0.02).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
plate = np.where(near[..., None], plate, art)
# Huang's colour, premultiplied so that at rest he recombines with the plate to exactly the art
JC = art - plate * (1 - FG[..., None])

# --- the red block, and the face under it
r, g, b = plate[..., 0], plate[..., 1], plate[..., 2]
RED = np.clip((r - np.maximum(g, b) - 15) / 45, 0, 1)                   # soft: takes the anti-aliased edges too
red_core = cv2.erode((RED > 0.95).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
face_px = (RED < 0.01) & (FG < 0.02)
# the face under the red. The red has its own coarser halftone, so only the smooth tone is taken from it (its red
# channel carries the face's light and dark), matched to the face's tonal range; then the face's own halftone screen
# (a 45-degree dot grid, period 15.75 px along both diagonals, fitted on the visible face) is laid back over it.
Lp = plate.mean(2)
q = np.linspace(0, 100, 201)
rb_ = cv2.GaussianBlur(r, (0, 0), 3.5)
src_q = np.percentile(rb_[red_core], q); dst_q = np.percentile(cv2.GaussianBlur(Lp, (0, 0), 3.5)[face_px], q)
tone = np.interp(rb_, src_q, dst_q).astype(np.float32)
yy_, xx_ = np.mgrid[0:H, 0:W].astype(np.float32)
AMP_T = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 255]
AMP_A = [1.5, 2.6, 4.0, 5.9, 8.8, 11.7, 13.0, 11.4, 12.1, 11.9, 6.0, 1.0]   # dot strength by tone, per diagonal
amp = np.interp(tone, AMP_T, AMP_A).astype(np.float32) / 2               # measured as 2x the cosine amplitude
screen = amp * (np.cos(2 * np.pi * (xx_ + yy_) / 15.75 - 1.68) + np.cos(2 * np.pi * (xx_ - yy_) / 15.75 + 1.79))
# tone correction: at every letter edge, compare the face just outside with the rebuild just inside, and spread that
# difference smoothly through each red area, so the rebuilt face meets the real one without a step
def nblur(v, w, sig):
    return cv2.GaussianBlur(v * w, (0, 0), sig) / np.maximum(cv2.GaussianBlur(w, (0, 0), sig), 1e-4)
redb = (RED > 0.5).astype(np.uint8)
outer = ((cv2.dilate(redb, np.ones((13, 13), np.uint8)) > 0) & (RED < 0.01) & (FG < 0.02)).astype(np.float32)
inner = ((redb > 0) & ~(cv2.erode(redb, np.ones((13, 13), np.uint8)) > 0) & (FG < 0.02)).astype(np.float32)
Lout = nblur(cv2.GaussianBlur(Lp, (0, 0), 3.5), outer, 8); Lin = nblur(tone, inner, 8)
both = (cv2.GaussianBlur(outer, (0, 0), 8) > 0.02) & (cv2.GaussianBlur(inner, (0, 0), 8) > 0.02)
dw = (both & (inner > 0)).astype(np.float32)
corr = nblur((Lout - Lin).astype(np.float32), dw, 25) * 0.5 + nblur((Lout - Lin).astype(np.float32), dw, 70) * 0.5
tone = np.clip(tone + corr, 0, 255)
# just around the letter edges (half-red pixels) take the tone from the real face beside them instead
face_tone = nblur(cv2.GaussianBlur(Lp, (0, 0), 3.5), ((RED < 0.01) & (FG < 0.02)).astype(np.float32), 4)
tone = np.where(RED > 0.9, tone, face_tone).astype(np.float32)
amp = np.interp(tone, AMP_T, AMP_A).astype(np.float32) / 2
screen = amp * (np.cos(2 * np.pi * (xx_ + yy_) / 15.75 - 1.68) + np.cos(2 * np.pi * (xx_ - yy_) / 15.75 + 1.79))
# plus the face's own irregular grain, borrowed from a plain patch of cheek (dot grid removed) and tiled
_lo = cv2.GaussianBlur(Lp, (0, 0), 3.5)
_amp = np.interp(_lo, AMP_T, AMP_A).astype(np.float32) / 2
_res = Lp - _lo - _amp * (np.cos(2 * np.pi * (xx_ + yy_) / 15.75 - 1.68) + np.cos(2 * np.pi * (xx_ - yy_) / 15.75 + 1.79))
_res = _res / np.maximum(_amp, 1.0)                                       # grain per unit of dot strength
tile = _res[300:900, 1450:1890]
th_, tw_ = tile.shape
# mirror-tile so the joins don't show
tile = np.block([[tile, tile[:, ::-1]], [tile[::-1], tile[::-1, ::-1]]])
grain = np.tile(tile, (H // (2 * th_) + 1, W // (2 * tw_) + 1))[:H, :W]
Lrec = np.clip(tone + screen + 0.9 * grain * amp, 0, 255)
fits_f = [np.polyfit(Lp[face_px][::50], plate[..., c][face_px][::50], 1) for c in range(3)]
face_rec = np.clip(np.dstack([np.polyval(fits_f[c], Lrec) for c in range(3)]), 0, 255).astype(np.float32)
fr_ = Lrec
WREC = cv2.GaussianBlur(cv2.dilate((RED > 0.02).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32), (0, 0), 1.5)
WREC = np.maximum(WREC, RED)
face = plate * (1 - WREC[..., None]) + face_rec * WREC[..., None]       # rebuilt a few px past each letter edge, feathered
# the red over the face, per channel: fitted on the solid red (used only while a letter is moving)
fits = [np.polyfit(fr_[red_core][::20], plate[..., c][red_core][::20], 1) for c in range(3)]
model_red = np.clip(np.dstack([np.polyval(fits[c], face[..., 0]) for c in range(3)]), 0, 255).astype(np.float32)

# --- letter cells: F | A | F | O, split at the red gaps between the letters (soft 12px edges)
EDGES = [480, 1000, 1570]
def cellw(k):
    lo = -1e9 if k == 0 else EDGES[k - 1]; hi = 1e9 if k == 3 else EDGES[k]
    return (np.clip((xx - lo) / 12 + 0.5, 0, 1) * np.clip((hi - xx) / 12 + 0.5, 0, 1)).repeat(H, 0)
CELLS = [cellw(k) for k in range(4)]
CENTRES = [(350, 540), (740, 540), (1290, 540), (1700, 540)]
STAMP = [0.2, 0.42, 0.64, 0.86]                                         # F, A, F, O land
LEAVE = [9.36, 9.24, 9.12, 9.0]                                         # O leaves first

def ease_back(u, k=1.6): u -= 1; return 1 + (k + 1) * u ** 3 + k * u ** 2
def cell_state(k, t):
    """(alpha, scale, flash) for letter-cell k at time t."""
    s = (t - STAMP[k]) / 0.22
    if s < 0: return 0.0, 1.0, 0.0
    a = min(1.0, s / 0.35)
    sc = 1.0 + 0.16 * (1 - ease_back(min(1.0, s)))                    # 1.16 -> 1 (tiny overshoot)
    fl = max(0.0, 1 - (t - STAMP[k]) / 0.3) ** 2
    o = np.clip((t - LEAVE[k]) / 0.22, 0, 1)
    return a * (1 - o) ** 2, sc + 0.06 * o, fl

def background(t):
    out = face.copy()
    for k in range(4):
        a, sc, fl = cell_state(k, t)
        if a <= 0.001: continue
        w = CELLS[k]
        if sc == 1.0 and fl == 0 and a >= 1:                          # settled: exactly the art's red
            out = out * (1 - w[..., None]) + plate * w[..., None]
            continue
        cx, cy = CENTRES[k]
        M = np.float32([[sc, 0, (1 - sc) * cx], [0, sc, (1 - sc) * cy]])
        A = cv2.warpAffine(RED * w, M, (W, H), flags=cv2.INTER_LINEAR)[..., None] * a
        hot = model_red + fl * np.array([110, 70, 70], np.float32) * (A > 0)   # flash: red-hot as it lands
        out = out * (1 - A) + np.clip(hot, 0, 255) * A
    return out

CX, CY = 0.5 * W, 0.45 * H
def cam(t, depth):
    e = 0.5 - 0.5 * math.cos(TAU * t / P)
    z = 1 + 0.07 * e * depth
    return np.float32([[z, 0, (1 - z) * CX], [0, z, (1 - z) * CY]])

def frame(t):
    bgw = cv2.warpAffine(background(t), cam(t, 0.35), (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    Mf = cam(t, 1.0)
    fa = cv2.warpAffine(FG, Mf, (W, H), flags=cv2.INTER_LINEAR)[..., None]
    fc = cv2.warpAffine(JC, Mf, (W, H), flags=cv2.INTER_CUBIC)
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
