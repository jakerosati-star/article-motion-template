"""Miller Daddy, "good boy" variant: the slow push-in (animate_art.py) plus a comic speech bubble from Miller,
"Good boy.", that pops in at 0.35s, holds, and pops away before the loop point (frame 0 = the art).
Run from art_build/. Writes ../art_loop_goodboy.mp4"""
import math, os, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from render_card import font

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
src = np.array(Image.open('../art_still.jpg').convert('RGB'))
s = W / src.shape[1]
CX, CY = 0.5 * W, 0.45 * H; ZOOM = 0.06

# ---- bubble (drawn 4x, in 1920 art px) ----
SS = 4
BX0, BY0, BX1, BY1 = 1205, 100, 1615, 280          # ellipse box, to the right of Miller's head, clear of the 9:16 button zone even at full push-in
TIP = (1165, 300)                                  # tail tip, just off his cheek by the mouth
layer = Image.new('L', (W * SS, H * SS), 0); d = ImageDraw.Draw(layer)
d.ellipse([v * SS for v in (BX0, BY0, BX1, BY1)], fill=255)
d.polygon([(TIP[0] * SS, TIP[1] * SS), (1250 * SS, 212 * SS), (1305 * SS, 260 * SS)], fill=255)
fill_m = layer.resize((W, H), Image.LANCZOS)
outline_m = layer.filter(ImageFilter.MaxFilter(25)).resize((W, H), Image.LANCZOS)   # ~6px black stroke
txt = Image.new('L', (W * SS, H * SS), 0); f = font(700, 75, 88 * SS)
label = 'Good boy.'; bb = f.getbbox(label)
cx, cy = (BX0 + BX1) / 2 * SS, (BY0 + BY1) / 2 * SS
ImageDraw.Draw(txt).text((cx - (bb[0] + bb[2]) / 2, cy - (bb[1] + bb[3]) / 2), label, font=f, fill=255)
txt_m = txt.resize((W, H), Image.LANCZOS)
o = np.array(outline_m, np.float32) / 255; fm = np.array(fill_m, np.float32) / 255; tm = np.array(txt_m, np.float32) / 255
col = np.zeros((H, W, 3), np.float32) + 255 * fm[..., None] * (1 - tm[..., None])   # white fill, black text, black stroke
# soft drop shadow
sh = cv2.GaussianBlur(o, (0, 0), 7); sh = np.roll(np.roll(sh, 8, 0), 5, 1) * 0.45
alpha = np.clip(o + sh * (1 - o), 0, 1)
premul = np.clip(col, 0, 255).astype(np.float32)   # premultiplied: white scaled by fill coverage; stroke and shadow are black
A = alpha.astype(np.float32)

def ease_back(u, k=1.9): u -= 1; return 1 + (k + 1) * u ** 3 + k * u ** 2
def bubble_scale(t):
    if t < 0.35: return 0
    if t < 0.75: return ease_back((t - 0.35) / 0.4)
    if t < 8.35: return 1
    if t < 8.6: u = (t - 8.35) / 0.25; return 1 - u ** 3 if u < 1 else 0
    return 0

enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', '-preset', 'slow', '-movflags', '+faststart', '../art_loop_goodboy.mp4'],
                       stdin=subprocess.PIPE)
for i in range(N):
    t = i / FPS
    z = 1 + ZOOM * (0.5 - 0.5 * math.cos(TAU * i / N))
    M = np.float32([[s * z, 0, (1 - z) * CX], [0, s * z, (1 - z) * CY]])
    fr = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    b = bubble_scale(t)
    if b > 0.001:
        # pop from the tail tip, then ride the same push-in as the art
        Mb = np.float32([[b * z, 0, (1 - b) * TIP[0] * z + (1 - z) * CX], [0, b * z, (1 - b) * TIP[1] * z + (1 - z) * CY]])
        pc = cv2.warpAffine(premul, Mb, (W, H), flags=cv2.INTER_LINEAR)
        pa = cv2.warpAffine(A, Mb, (W, H), flags=cv2.INTER_LINEAR)[..., None]
        fr = fr * (1 - pa) + pc
    out = np.clip(fr, 0, 255).astype(np.uint8)
    enc.stdin.write(out.tobytes())
enc.stdin.close(); enc.wait()
