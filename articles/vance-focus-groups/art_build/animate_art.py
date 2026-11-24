"""Vance Focus Groups art: cartoon tears and laughter over the untouched art. Tears well up along Vance's lower lids
and drip down his cheeks one after another (from 0.2s), leaving wet streaks; beside him, "HA!" "HA!" "HA!" pops out
of the laughing man in bursts and floats up, wobbling. Slow push-in. Frame 0 = the art; everything has dripped away
or faded by the loop point. Run from art_build/. Writes ../art_loop.mp4"""
import math, os, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
art = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)
FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'fonts', 'SpecialGothic-VariableFont_wdthwght.ttf')
SS = 2                                                     # drawn at 2x, then scaled down (smooth edges)

# tear paths: from the lower lid down the cheek into the beard (Vance's left and right eye)
PATHS = [[(714, 300), (708, 345), (700, 395), (694, 450)], [(866, 296), (872, 342), (880, 392), (886, 448)]]
LIDS = [((714, 300), 46), ((866, 296), 46)]
def along(path, u):
    u = min(max(u, 0), 1) * (len(path) - 1); i = min(int(u), len(path) - 2); f = u - i
    (x0, y0), (x1, y1) = path[i], path[i + 1]; return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
# drops: (eye, start time, size)
DROPS = [(0, 0.35, 20), (1, 0.6, 19), (0, 1.5, 17), (1, 2.0, 20), (0, 3.0, 18), (1, 3.6, 17),
         (0, 4.9, 20), (1, 5.2, 19), (0, 6.3, 17), (1, 6.9, 19), (0, 7.8, 18)]
DROP_T = 1.1
def cry(t):                                                 # how wet the eyes are
    return min(1, max(0, (t - 0.15) / 0.3)) * min(1, max(0, (P - 0.5 - t) / 0.6))

HA_POS = [(1720, 330), (1800, 220), (1690, 120), (1760, 380), (1830, 260), (1700, 160)]   # in the blue, up and right of his face
HAS = [(0.3, 0, -12, 92), (0.62, 1, 10, 104), (0.95, 2, -6, 118), (4.4, 3, 8, 96), (4.7, 4, -10, 108), (5.0, 5, 6, 120)]
HA_LIFE = 1.6
def font(size):
    f = ImageFont.truetype(FONT, size); f.set_variation_by_axes([700, 75]); return f

def overlay(t):
    img = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    s = SS
    c = cry(t)
    if c > 0:
        for (x, y), rx in LIDS:                             # tears welling along the lower lid
            d.ellipse([(x - rx) * s, (y - 7) * s, (x + rx) * s, (y + 7) * s], fill=(170, 220, 255, int(150 * c)))
            d.ellipse([(x - rx * 0.6) * s, (y - 3) * s, (x + rx * 0.2) * s, (y + 1) * s], fill=(255, 255, 255, int(170 * c)))
    for eye, t0, r in DROPS:
        u = (t - t0) / DROP_T
        if u < 0 or u > 1.25: continue
        path = PATHS[eye]
        # the wet streak it leaves (fades after)
        a_st = int(90 * min(1, u * 3) * max(0, 1 - max(0, u - 0.9) / 0.35))
        pts = [along(path, k / 20 * min(u, 1)) for k in range(21)]
        if a_st > 0:
            d.line([(x * s, y * s) for x, y in pts], fill=(190, 230, 255, a_st), width=int(r * 0.7 * s), joint='curve')
        if u <= 1:
            x, y = along(path, u ** 1.4)                   # starts slow, speeds up
            grow = min(1, u * 5)
            rr = r * (0.5 + 0.5 * grow)
            a = int(255 * min(1, (1 - u) * 6))
            # teardrop: round body with a point upward, cartoon outline and highlight
            body = [(x - rr) * s, (y - rr) * s, (x + rr) * s, (y + rr) * s]
            tip = [(x - rr * 0.82) * s, (y - rr * 0.3) * s, x * s, (y - rr * 2.1) * s, (x + rr * 0.82) * s, (y - rr * 0.3) * s]
            d.polygon(tip, fill=(40, 120, 220, a)); d.ellipse(body, fill=(40, 120, 220, a))
            body2 = [(x - rr * 0.78) * s, (y - rr * 0.78) * s, (x + rr * 0.78) * s, (y + rr * 0.78) * s]
            tip2 = [(x - rr * 0.6) * s, (y - rr * 0.2) * s, x * s, (y - rr * 1.7) * s, (x + rr * 0.6) * s, (y - rr * 0.2) * s]
            d.polygon(tip2, fill=(120, 200, 255, a)); d.ellipse(body2, fill=(120, 200, 255, a))
            d.ellipse([(x - rr * 0.45) * s, (y - rr * 0.6) * s, (x - rr * 0.05) * s, (y - rr * 0.15) * s], fill=(255, 255, 255, a))
    for t0, k, rot, size in HAS:
        u = (t - t0) / HA_LIFE
        if not 0 <= u < 1: continue
        pop = 1 + 0.35 * math.exp(-u * 14) * math.sin(u * 40) if u < 0.25 else 1
        sc = min(1, u * 8) * pop
        a = int(255 * min(1, (1 - u) * 3))
        x, y = HA_POS[k]; y -= 70 * u; x += 12 * math.sin(u * 9)
        fsz = max(4, int(size * sc * s))
        txt = Image.new('RGBA', (fsz * 3, fsz * 2), (0, 0, 0, 0)); td = ImageDraw.Draw(txt)
        td.text((fsz * 0.3, fsz * 0.2), 'HA!', font=font(fsz), fill=(255, 255, 255, a), stroke_width=max(2, fsz // 12), stroke_fill=(10, 10, 10, a))
        txt = txt.rotate(rot + 6 * math.sin(u * 12), resample=Image.BICUBIC, expand=False)
        img.alpha_composite(txt, (int(x * s - txt.width / 2), int(y * s - txt.height / 2)))
    o = np.array(img.resize((W, H), Image.LANCZOS)).astype(np.float32)
    return o[..., :3], o[..., 3:] / 255

def frame(t):
    rgb, a = overlay(t)
    f = art * (1 - a) + rgb * a
    e = 0.5 - 0.5 * math.cos(TAU * t / P); z = 1 + 0.05 * e
    Mz = np.float32([[z, 0, (1 - z) * 800], [0, z, (1 - z) * 330]])
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
