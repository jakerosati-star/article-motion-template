"""MAGA Preps art: mushroom clouds churn and flicker, embers rise, the top grasshopper twitches,
the ballot drop box stays put. Seamless 10s loop, full image from frame 0. Writes ../art_loop.mp4"""
import math, random, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw
from scipy import ndimage

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32)

# --- layers -------------------------------------------------------------------
BOX = (560, 622, 1194, 1080)                                   # the drop box (static anchor)
box_m = np.zeros((H, W), np.float32); box_m[BOX[1]:, BOX[0]:BOX[2] + 1] = 1
def crop_mask(name, x0, y0):
    m = np.zeros((H, W), np.float32); c = np.array(Image.open(name)).astype(np.float32) / 255
    m[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] = c; return m
gh1 = crop_mask('gh1_mask.png', 420, 465)                      # top grasshopper (moves)
gh2 = crop_mask('gh2_mask.png', 950, 595)                      # side grasshopper (static, part of the box layer)
solid = ndimage.binary_fill_holes(ndimage.binary_closing(gh1 > 0.2, structure=np.ones((9, 9)), iterations=2))
belly = Image.new('L', (W, H), 0)          # the cutout missed the striped belly under the wing: traced by hand
ImageDraw.Draw(belly).polygon([(515, 598), (516, 609), (540, 618), (580, 625), (620, 629), (650, 630), (680, 626), (720, 622), (765, 618), (768, 598)], fill=255)
solid = ndimage.binary_fill_holes(solid | (np.array(belly) > 0))
gh1 = np.clip(ndimage.gaussian_filter(solid.astype(np.float32), 0.8) * 1.15, 0, 1)   # solid body with a clean soft edge (no see-through belly)

# static foreground = box + side grasshopper; the top grasshopper's footprint on the box is filled in
fg_static = np.maximum(box_m, gh2)
fixed = cv2.inpaint(a.astype(np.uint8), cv2.dilate((gh1 > 0.1).astype(np.uint8), np.ones((7, 7), np.uint8)), 6, cv2.INPAINT_TELEA).astype(np.float32)
fg_rgb = fixed.copy()
# background (sky + clouds) with everything in front filled in; only its edges ever show
hole = cv2.dilate(((fg_static + gh1) > 0.05).astype(np.uint8), np.ones((15, 15), np.uint8))
bg = cv2.inpaint(a.astype(np.uint8), hole, 15, cv2.INPAINT_TELEA).astype(np.float32)
near = cv2.dilate(((fg_static > 0.5) | (gh1 > 0.02)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
bg = np.where(near[..., None], bg, a)          # keep the original everywhere the foreground doesn't actually cover
bg_lum = bg.mean(2) / 255
fire = np.clip((bg_lum - 0.35) / 0.5, 0, 1)                    # where the fire is bright

gh_rgba = np.dstack([a, gh1 * 255]).astype(np.float32)
# churn strength fades to zero next to the box and grasshoppers, so filled-in pixels never get dragged into view
dist = ndimage.distance_transform_edt(~near)
calm = np.clip((dist - 4) / 70, 0, 1).astype(np.float32)
GH_PIVOT = (690, 638)                                          # roughly where its legs meet the box top

# --- periodic noise for the billowing ----------------------------------------
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
rng = np.random.default_rng(5)
waves = [(rng.uniform(0.004, 0.012), rng.uniform(0, TAU), rng.uniform(0, TAU), rng.integers(1, 3), rng.uniform(0, TAU)) for _ in range(6)]
def field(t, axis):
    out = np.zeros((H, W), np.float32)
    for j, (f_, ang, ph, kk, ph2) in enumerate(waves):
        ang2 = ang + axis * 1.3
        out += np.sin((xx * math.cos(ang2) + yy * math.sin(ang2)) * f_ + ph + TAU * kk * t / P + ph2 * axis)
    return out / len(waves)

# embers (periodic): rise from the fire, drift, twinkle
er = random.Random(9); Ly = H + 120
embers = [dict(x=er.uniform(0, W), y0=er.uniform(0, Ly), n=er.choice([1, 2, 2, 3]), sw=er.uniform(8, 30), m=er.randint(1, 3),
               ph=er.uniform(0, TAU), r=er.choice([1.5, 2, 2, 2.5, 3, 4]), k=er.randint(2, 7)) for _ in range(90)]

def clamp(x): return max(0.0, min(1.0, x))
def over(base, rgb, al): return base * (1 - al[..., None]) + rgb * al[..., None]
def wv(k, t, ph=0): return math.sin(TAU * k * t / P + ph)

# grasshopper twitches: (time, duration, dx, dy, deg)
TW = [(1.6, 0.22, 2, -6, -2.0), (1.9, 0.18, -1, -3, 1.2), (5.2, 0.25, 4, -8, -2.5), (7.8, 0.2, -2, -4, 1.5), (8.05, 0.16, 1, -3, -1.0)]
def gh_pose(t):
    dx = dy = ang = 0.0
    for t0, d, x_, y_, g_ in TW:
        u = (t - t0) / d
        if 0 <= u <= 1: s = math.sin(math.pi * u); dx += x_ * s; dy += y_ * s; ang += g_ * s
    return dx, dy, ang + 0.3 * wv(2, t)

enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                       stdin=subprocess.PIPE)
for i in range(N):
    t = i / FPS
    # clouds churn: smooth displacement, stronger in the fire, plus a slow periodic swell
    dxm = (6 * field(t, 0) * (0.4 + 0.6 * fire)).astype(np.float32)
    dym = (6 * field(t, 1) * (0.4 + 0.6 * fire) - 3 * (0.5 - 0.5 * math.cos(TAU * t / P)) * fire).astype(np.float32)
    zb = 1.0 + 0.02 * (0.5 - 0.5 * math.cos(TAU * t / P))
    mx = xx + ((xx - W / 2) / zb + W / 2 + dxm - xx) * calm; my = yy + ((yy - H * 0.35) / zb + H * 0.35 + dym - yy) * calm
    f = cv2.remap(bg, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # fire flicker: brightness breathing in the bright areas
    glow = 1 + (0.05 * wv(9, t, 0.5) + 0.04 * wv(4, t, 2) + 0.03 * wv(13, t, 1.3)) * fire
    f = np.clip(f * glow[..., None], 0, 255)
    # embers
    el = np.zeros((H, W), np.float32)
    for e in embers:
        y = Ly - ((e['y0'] + Ly * e['n'] * t / P) % Ly) - 60
        x = e['x'] + e['sw'] * wv(e['m'], t, e['ph'])
        al = 0.55 + 0.45 * wv(e['k'], t, e['ph'])
        cv2.circle(el, (int(x * 4), int(y * 4)), int(e['r'] * 4), al, -1, lineType=cv2.LINE_AA, shift=2)
    elg = cv2.GaussianBlur(el, (0, 0), 3) * 0.9 + el
    f = np.clip(f + elg[..., None] * np.array([255, 170, 60], np.float32), 0, 255)
    # box (static) + side grasshopper
    f = over(f, fg_rgb, fg_static)
    # top grasshopper twitches
    dx, dy, ang = gh_pose(t)
    M = cv2.getRotationMatrix2D(GH_PIVOT, ang, 1.0); M[0, 2] += dx; M[1, 2] += dy
    g = cv2.warpAffine(gh_rgba, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    f = over(f, g[..., :3], g[..., 3] / 255)
    # embers drift in front of the box too, faintly
    f = np.clip(f + 0.35 * elg[..., None] * np.array([255, 170, 60], np.float32) * box_m[..., None], 0, 255)
    # camera push
    z = 1 + 0.03 * (0.5 - 0.5 * math.cos(TAU * t / P))
    Mc = np.array([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H * 0.55]], np.float32)
    out = cv2.warpAffine(f.astype(np.uint8), Mc, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if i in (0, 48, 51, 160, 240, 299): Image.fromarray(out).save(f'_f{i:03d}.png')
    enc.stdin.write(out.tobytes())
enc.stdin.close(); enc.wait()
