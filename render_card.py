#!/usr/bin/env python3
"""Render looping article cards (art loop + animated headline panel) in several aspect ratios.

Usage:  python3 render_card.py example/article.json [--formats 1x1,9x16]

The art loop is any seamless-looping 16:9 video (see README for how to make one).
The card loop length matches the art loop length.
"""
import json, math, os, subprocess, sys, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, 'fonts', 'SpecialGothic-VariableFont_wdthwght.ttf')
FPS = 30

# ---- Layouts ---------------------------------------------------------------
# art: (x, y, w, h) box the art loop is scaled into (keep 16:9).
# text: left x, column width, top y (None = vertically centre in the space below/beside art).
# head: headline font size. Everything else in the panel scales from it.
LAYOUTS = {
    '1x1':  dict(size=(1080, 1080), art=(0, 0, 1080, 608),    text_x=58, text_w=964, text_top=648,  text_bottom=1080, head=82),
    '4x5':  dict(size=(1080, 1350), art=(0, 0, 1080, 608),    text_x=64, text_w=952, text_top=650,  text_bottom=1350, head=98, center=True),
    '9x16': dict(size=(1080, 1920), art=(0, 0, 1080, 608),    text_x=64, text_w=952, text_top=0,    text_bottom=0,    head=112, group=(250, 1600)),
    '16x9': dict(size=(1920, 1080), art=(704, 202, 1216, 684), text_x=80, text_w=560, text_top=90,  text_bottom=990,  head=96, center=True, seam='left'),
}

# ---- Easing ----------------------------------------------------------------
def clamp(x): return max(0.0, min(1.0, x))
def eoc(t): return 1 - (1 - t) ** 3
def eoq(t): return 1 - (1 - t) ** 5
def eic(t): return t ** 3
def eob(t, s=2.2): t -= 1; return 1 + (s + 1) * t ** 3 + s * t ** 2


def font(weight, width, size):
    f = ImageFont.truetype(FONT, size)
    f.set_variation_by_axes([weight, width])
    return f


def text_img(txt, f, fill):
    bb = f.getbbox(txt)
    im = Image.new('RGBA', (bb[2] + 4, bb[3] + 8), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((0, 0), txt, font=f, fill=fill)
    return im


def wrap(txt, f, width):
    lines, cur = [], ''
    for w in txt.split():
        trial = (cur + ' ' + w).strip()
        if cur and f.getlength(trial) > width:
            lines.append(cur); cur = w
        else:
            cur = trial
    return lines + [cur]


def balanced_wrap(txt, f, width):
    """Greedy wrap; if that leaves one word alone on the last line, even out the line lengths instead."""
    greedy = wrap(txt, f, width)
    if len(greedy) < 2 or len(greedy[-1].split()) > 1: return greedy
    n = len(greedy); lo, hi = width * 0.4, width
    for _ in range(20):
        mid = (lo + hi) / 2
        if len(wrap(txt, f, mid)) > n: lo = mid
        else: hi = mid
    return wrap(txt, f, hi)


def fade(im, a):
    if a >= 1: return im
    arr = np.array(im); arr[:, :, 3] = (arr[:, :, 3] * a).astype(np.uint8)
    return Image.fromarray(arr)


def hex2rgb(h):
    h = h.lstrip('#'); return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def probe_duration(path):
    out = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path])
    return float(out)


def render(cfg, base, fmt, out_path):
    L = LAYOUTS[fmt]
    CW, CH = L['size']
    ax, ay, aw, ah = L['art']
    k = L['head'] / 82.0                      # scale factor relative to the 1x1 design
    RED = hex2rgb(cfg.get('accent', '#E4002B'))
    X, TW = L['text_x'], L['text_w']

    fH = font(700, 75, L['head']); lh = int(L['head'] * 1.0)
    fTag = font(700, 75, round(30 * k)); fSub = font(600, 100, round(27 * k)); fBy = font(700, 90, round(18 * k))
    AV = round(50 * k)

    # --- build text pieces & measure block height
    tagtxt = text_img(cfg['tag'], fTag, 'white'); tw = tagtxt.width + round(24 * k); th = round(44 * k)
    hlines = balanced_wrap(cfg['headline'], fH, TW)
    sublines = wrap(cfg.get('subhead', ''), fSub, TW) if cfg.get('subhead') else []
    sub_lh = round(36 * k)
    gap_tag = round(24 * k); gap_sub = round(24 * k); gap_by = round(24 * k)
    block_h = th + gap_tag + len(hlines) * lh + (gap_sub + len(sublines) * sub_lh if sublines else 0) + gap_by + AV
    top = L['text_top']
    if L.get('group'):   # centre art + text together inside a safe zone (vertical formats)
        g0, g1 = L['group']; gap = round(44 * k)
        ay = g0 + max(0, (g1 - g0 - (ah + gap + block_h)) // 2)
        top = ay + ah + gap
    elif L.get('center'):
        top = top + max(0, (L['text_bottom'] - L['text_top'] - block_h) // 2)
    tag_y = top
    head_y = tag_y + th + gap_tag
    sub_y = head_y + len(hlines) * lh + gap_sub
    by_y = (sub_y + len(sublines) * sub_lh if sublines else head_y + len(hlines) * lh) + gap_by

    words = []
    for li, ln in enumerate(hlines):
        x = X
        for w in ln.split(' '):
            bb = fH.getbbox(w)
            im = Image.new('RGBA', (bb[2] + 4, lh + 30), (0, 0, 0, 0))
            ImageDraw.Draw(im).text((0, 0), w, font=fH, fill='white')
            words.append((im, x, head_y + li * lh))
            x += fH.getlength(w + ' ')
    NW = len(words)
    subs = [text_img(s, fSub, (215, 215, 215)) for s in sublines]
    by1 = text_img(cfg['author'], fBy, (235, 235, 235)); by2 = text_img(cfg['date'], fBy, (200, 200, 200))

    av = None
    if cfg.get('avatar'):
        big = AV * 4
        ph = Image.open(os.path.join(base, cfg['avatar'])).convert('RGB')
        s = min(ph.size); ph = ph.crop(((ph.width - s) // 2, (ph.height - s) // 2, (ph.width + s) // 2, (ph.height + s) // 2)).resize((big, big), Image.LANCZOS)
        m = Image.new('L', (big, big), 0); ImageDraw.Draw(m).ellipse((0, 0, big - 1, big - 1), fill=255)
        av = Image.new('RGBA', (big, big), (0, 0, 0, 0)); av.paste(ph, (0, 0), m)
        ImageDraw.Draw(av).ellipse((2, 2, big - 3, big - 3), outline=(255, 255, 255, 255), width=7)
        av = av.resize((AV, AV), Image.LANCZOS)
    by_x = X + (AV + round(14 * k) if av else 0)

    art_path = os.path.join(base, cfg['art_loop'])
    P = probe_duration(art_path); N = int(round(P * FPS))
    E = P - 10.0   # timings below were designed on a 10s loop; outro is anchored to the loop end

    dec = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-i', art_path, '-vf', f'scale={aw}:{ah}:flags=lanczos',
                            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
    enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{CW}x{CH}',
                            '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'slow',
                            '-movflags', '+faststart', out_path], stdin=subprocess.PIPE)
    seam = L.get('seam', 'bottom'); sw = max(4, round(5 * k))
    for i in range(N):
        t = i / FPS
        raw = dec.stdout.read(aw * ah * 3)
        if len(raw) < aw * ah * 3: break
        fr = Image.new('RGBA', (CW, CH), (0, 0, 0, 255))
        fr.paste(Image.frombytes('RGB', (aw, ah), raw), (ax, ay))
        d = ImageDraw.Draw(fr)

        # accent seam: wipes in, retracts before the headline leaves
        p = eoq(clamp((t - 0.6) / 0.7)); p0 = eic(clamp((t - 9.0 - E) / 0.45))
        if p > p0:
            if seam == 'bottom':
                d.rectangle((ax + int(aw * p0), ay + ah, ax + int(aw * p), ay + ah + sw), fill=RED)
            else:  # vertical line on the art's left edge
                d.rectangle((ax - sw, ay + int(ah * p0), ax - 1, ay + int(ah * p)), fill=RED)

        # tag box
        p = eoq(clamp((t - 0.8) / 0.45)) * (1 - eic(clamp((t - 8.85 - E) / 0.4)))
        if p > 0:
            d.rectangle((X, tag_y, X + int(tw * p), tag_y + th), fill=RED)
            q = eoc(clamp((t - 1.0) / 0.4))
            if q > 0:
                box = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
                box.alpha_composite(fade(tagtxt, q), (round(12 * k), int(round(4 * k) + (1 - q) * 20 * k)))
                fr.alpha_composite(box.crop((0, 0, max(1, int(tw * p)), th)), (X, tag_y))

        # headline: first in, last out (words rise from / sink into a mask)
        for j, (im, x, y) in enumerate(words):
            q = eoq(clamp((t - 0.05 - j * 0.05) / 0.55))
            qo = eic(clamp((t - (P - 0.36 - (NW - 1) * 0.03) - (NW - 1 - j) * 0.03) / 0.28))
            if q <= 0 or qo >= 1: continue
            off = int((1 - q) * (lh + 10) + qo * (lh + 10))
            band = Image.new('RGBA', (im.width, lh + round(14 * k)), (0, 0, 0, 0)); band.alpha_composite(im, (0, off))
            fr.alpha_composite(band, (int(x), y - round(4 * k)))

        # subhead + byline
        items = [(s, (X, sub_y + n * sub_lh), 1.2 + n * 0.1, 8.65 + E) for n, s in enumerate(subs)]
        items += [(by1, (by_x, by_y + round(3 * k)), 1.65, 8.5 + E), (by2, (by_x, by_y + round(27 * k)), 1.75, 8.5 + E)]
        for im, (px, py), st, so in items:
            q = eoc(clamp((t - st) / 0.6)); qo = clamp((t - so) / 0.3)
            if q > 0 and qo < 1:
                fr.alpha_composite(fade(im, q * (1 - qo)), (px, int(py + (1 - q) * 18 * k + qo * 10 * k)))
        if av is not None:
            q = clamp((t - 1.5) / 0.5); qo = clamp((t - 8.45 - E) / 0.3)
            if q > 0 and qo < 1:
                sz = max(1, int(AV * eob(q) * (1 - qo * qo)))
                fr.alpha_composite(fade(av.resize((sz, sz), Image.LANCZOS), min(1, q * 2.5)), (X + (AV - sz) // 2, by_y + (AV - sz) // 2))

        out = fr.convert('RGB')
        if i == int(5 * FPS):
            out.save(out_path.replace('.mp4', '_still.png'))
        enc.stdin.write(out.tobytes())
    enc.stdin.close(); enc.wait(); dec.stdout.close(); dec.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--formats', help='comma list, e.g. 1x1,9x16 (default: from config)')
    a = ap.parse_args()
    base = os.path.dirname(os.path.abspath(a.config))
    cfg = json.load(open(a.config))
    fmts = a.formats.split(',') if a.formats else cfg.get('formats', list(LAYOUTS))
    prefix = os.path.join(base, cfg.get('output_prefix', 'out/card'))
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    for f in fmts:
        out = f'{prefix}_{f}.mp4'
        print('rendering', out, flush=True)
        render(cfg, base, f, out)


if __name__ == '__main__':
    main()
