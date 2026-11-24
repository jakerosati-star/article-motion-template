"""Miller 2028 art: Miller (and his podium) steps forward from the scene: he scales up while everything behind him
(Vance and Rubio crossed out, the briefing room) scales down and dims a touch, at 0.1s, easing in over ~3s; he holds, then both slowly
ease back to the original framing by the loop point, so frame 0 = the art. The foreground is Miller (person matte) and
the podium (traced outline). Both scale about the bottom centre, so the
background only ever shrinks under the growing foreground and nothing behind Miller is exposed.
Run from art_build/. Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
art = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
# foreground: Miller from the person matte (soft edges kept), plus the podium as its traced outline (straight edges,
# anti-aliased); the seal is part of the podium
from rembg import remove, new_session
_m = np.array(remove(Image.open('still_1920.png').convert('RGB'), session=new_session('u2net_human_seg'), only_mask=True)).astype(np.float32) / 255
_x = np.arange(W)[None, :]
_b = ((_m > 0.5) & (_x > 500) & (_x < 1450)).astype(np.uint8)
_n, _l, _s, _ = cv2.connectedComponentsWithStats(_b, 8)
_keep = cv2.dilate((_l == 1 + int(np.argmax(_s[1:, 4]))).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
MILLER = _m * _keep
_S = 4; _pod = np.zeros((H * _S, W * _S), np.uint8)
cv2.fillPoly(_pod, [np.int32([(p_[0] * _S, p_[1] * _S) for p_ in [(301, 946), (442, 887), (1580, 887), (1749, 950), (1749, 1085), (301, 1085)]])], 255, cv2.LINE_AA)
POD = cv2.resize(_pod, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
FG = np.maximum(MILLER, POD)
# the background behind him, rebuilt in grey only (the background photo is black and white, so no navy or skin
# colour can bleed in), from a generous margin so none of his outline is left in it
hole = cv2.dilate((FG > 0.02).astype(np.uint8), np.ones((25, 25), np.uint8))
gray = art.mean(2)
src_g = np.where(hole > 0, 0, gray).astype(np.uint8)
fill = cv2.inpaint(src_g, hole * 255, 15, cv2.INPAINT_TELEA).astype(np.float32)
fill = cv2.GaussianBlur(fill, (0, 0), 3)
bg = np.where((hole > 0)[..., None], np.repeat(fill[..., None], 3, 2), art)
FGC = art * FG[..., None]
ANCHOR = (960.0, 1080.0)
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = np.clip((np.sqrt(((_xx - W / 2) / (W / 2)) ** 2 + ((_yy - H * 0.6) / (H * 0.75)) ** 2) - 0.55) / 0.5, 0, 1)

def ease(u): u = min(1, max(0, u)); return u * u * (3 - 2 * u)
def amt(t):                                   # 0 = original framing, 1 = fully separated
    return ease((t - 0.1) / 3.2) * (1 - ease((t - 7.3) / 2.5))      # slow grow over ~3s, slow return

def frame(t):
    a = amt(t)
    if a <= 0: return art.copy()
    sb = 1 - 0.035 * a; sf = 1 + 0.07 * a
    Mb = np.float32([[sb, 0, (1 - sb) * ANCHOR[0]], [0, sb, (1 - sb) * ANCHOR[1]]])
    Mf = np.float32([[sf, 0, (1 - sf) * ANCHOR[0]], [0, sf, (1 - sf) * ANCHOR[1]]])
    b = cv2.warpAffine(bg, Mb, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # the thin band the shrunk scene leaves at the edges: smeared edge pixels, sunk into a vignette as it recedes
    inside = cv2.GaussianBlur(cv2.warpAffine(np.ones((H, W), np.float32), Mb, (W, H)), (0, 0), 4)
    soft = np.repeat(cv2.GaussianBlur(b.mean(2), (0, 0), 12)[..., None], 3, 2) * 0.6   # grey, dark: no smeared red
    b = b * inside[..., None] + soft * (1 - inside[..., None])
    b = b * ((1 - 0.2 * a) * (1 - 0.55 * a * VIG))[..., None]
    fa = cv2.warpAffine(FG, Mf, (W, H))[..., None]; fc = cv2.warpAffine(FGC, Mf, (W, H), flags=cv2.INTER_CUBIC)
    sh = cv2.GaussianBlur(fa[..., 0], (0, 0), 25)[..., None] * 0.45 * a          # he casts a soft shadow on the scene
    b = b * (1 - sh * (1 - fa))
    return b * (1 - fa) + fc

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        for t in map(float, sys.argv[1:]): Image.fromarray(np.clip(frame(t), 0, 255).astype(np.uint8)).save(f'_t{t:.2f}.png')
        sys.exit()
    enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'], stdin=subprocess.PIPE)
    for i in range(N): enc.stdin.write(np.clip(frame(i / FPS), 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close(); enc.wait()
