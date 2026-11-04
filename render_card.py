#!/usr/bin/env python3
"""Render looping article cards (art loop + animated headline panel) in several aspect ratios.

Usage:  python3 render_card.py example/article.json [--formats 1x1,9x16]

The art loop is any seamless-looping 16:9 video (see README for how to make one).
The card loop length matches the art loop length.
"""
import json, math, os, re, subprocess, sys, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, 'fonts', 'SpecialGothic-VariableFont_wdthwght.ttf')
FPS = 30
TAU = 2 * math.pi

# ---- Layouts ---------------------------------------------------------------
# art: (x, y, w, h) box the art loop is scaled into (keep 16:9).
# text: left x, column width, top y (None = vertically centre in the space below/beside art).
# head: headline font size. Everything else in the panel scales from it.
LAYOUTS = {
    '1x1':  dict(size=(1080, 1080), art=(0, 0, 1080, 608),    text_x=58, text_w=964, text_top=648,  text_bottom=1080, head=82, bias=0.3),
    '4x5':  dict(size=(1080, 1350), art=(0, 0, 1080, 608),    text_x=64, text_w=952, text_top=650,  text_bottom=1350, head=98, center=True, bias=0.3),
    '9x16': dict(size=(1080, 1920), art=(0, 0, 1080, 608),    text_x=64, text_w=952, text_top=0,    text_bottom=0,    head=112, group=(250, 1600)),
    '16x9': dict(size=(1920, 1080), art=(704, 202, 1216, 684), text_x=80, text_w=560, text_top=90,  text_bottom=990,  head=96, center=True, seam='left'),
}

# Bulwark logo per format: which lockup, its height in px, and where it sits.
#   'row'    = right-aligned to the text column, vertically centred on the byline row
#   'bottom' = left-aligned with the text, bottom edge at the given y
LOGO_DIR = os.path.join(HERE, 'BW Logos (transparent)')
LOGO_FILES = {'horizontal': 'Horizontal Logo, White TRANSPARENT.png', 'stacked': 'Stacked Logo, White TRANSPARENT.png'}
LOGO_OPACITY = 0.7
LOGOS = {
    '1x1':  dict(kind='horizontal', h=40, at='row'),
    '4x5':  dict(kind='horizontal', h=44, at='row'),
    '9x16': dict(kind='stacked',    h=84, at='row'),
    '16x9': dict(kind='horizontal', h=46, at='bottom', y=990),
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
    f.spec = (weight, width, size)
    return f


SS = 4   # text is drawn at 4x and scaled down: avoids whole-pixel letter positions (uneven spacing) at small sizes


def text_img(txt, f, fill, height=None):
    """Render text supersampled. Returns an RGBA image whose (0,0) matches ImageDraw.text((0,0)) at the base size."""
    w, wd, size = f.spec
    big = font(w, wd, size * SS)
    bb = big.getbbox(txt)
    W = (bb[2] + SS * 4) // SS + 1
    H = height if height else (bb[3] + SS * 8) // SS + 1
    m = Image.new('L', (W * SS, H * SS), 0)
    ImageDraw.Draw(m).text((0, 0), txt, font=big, fill=255)
    m = m.resize((W, H), Image.LANCZOS)
    col = Image.new('RGBA', (W, H), fill if isinstance(fill, tuple) else ImageColor.getrgb(fill))
    col.putalpha(m)
    return col


def text_len(txt, f):
    w, wd, size = f.spec
    return font(w, wd, size * SS).getlength(txt) / SS




def wrap(txt, f, width):
    lines, cur = [], ''
    for w in txt.split():
        trial = (cur + ' ' + w).strip()
        if cur and text_len(trial, f) > width:
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


HEADSHOTS = os.path.join(HERE, 'headshots')


def slug(name):
    """'Jonathan V. Last' -> 'jonathan-v-last' (the headshots/ file naming convention)."""
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


def find_headshot(author, base):
    """An explicit "avatar" path wins (relative to the article folder); otherwise find the author's photo in headshots/.
    File names are matched loosely: 'Sam Stein .JPG', 'sam_stein.png' and 'sam-stein.jpg' all match "SAM STEIN".
    If there's no exact match, first + last name is enough ("JONATHAN V. LAST" also matches 'jonathan-last.jpg')."""
    if author.get('avatar'):
        for p in (os.path.join(base, author['avatar']), os.path.join(HEADSHOTS, author['avatar'])):
            if os.path.exists(p): return p
    want = slug(author['name']); parts = want.split('-')
    files = [f for f in sorted(os.listdir(HEADSHOTS)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    for f in files:
        if slug(os.path.splitext(f)[0]) == want: return os.path.join(HEADSHOTS, f)
    for f in files:
        have = slug(os.path.splitext(f)[0]).split('-')
        if have[0] == parts[0] and have[-1] == parts[-1]: return os.path.join(HEADSHOTS, f)
    print(f"  warning: no headshot for {author['name']!r} (expected headshots/{want}.jpg); rendering without a photo", flush=True)
    return None


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
    tagtxt = text_img(cfg.get('tag') or 'THE BULWARK', fTag, 'white'); tw = tagtxt.width + round(24 * k); th = round(44 * k)
    hlines = balanced_wrap(cfg['headline'], fH, TW)
    if len(hlines) == 1:   # short headline: scale it up (max 1.5x) while it still fits on one line
        hs = L['head']
        while hs < L['head'] * 1.5 and text_len(cfg['headline'], font(700, 75, hs + 2)) <= TW: hs += 2
        fH = font(700, 75, hs); lh = hs; hlines = [cfg['headline']]
    sublines = wrap(cfg.get('subhead', ''), fSub, TW) if cfg.get('subhead') else []
    sub_lh = round(36 * k)
    sub_col = (215, 215, 215)
    if cfg.get('subhead_prominent'):   # promos: the offer line is bigger, bolder and white
        fSub = font(700, 100, round(36 * k)); sub_lh = round(46 * k); sub_col = (255, 255, 255)
        sublines = balanced_wrap(cfg['subhead'], fSub, TW)
    # gap_tag = visible gap from the tag box to the top of the headline's capitals.
    # The font has built-in space above its capitals (bbox top of 'H'); subtract it, and add back the 4px the mask box is raised by.
    gap_tag = round(26 * k) - fH.getbbox('H')[1] + round(4 * k)
    gap_sub = round(24 * k); gap_by = round(24 * k)
    block_h = th + gap_tag + len(hlines) * lh + (gap_sub + len(sublines) * sub_lh if sublines else 0) + gap_by + AV
    top = L['text_top']
    if L.get('group'):   # centre art + text together inside a safe zone (vertical formats)
        g0, g1 = L['group']; gap = round(44 * k)
        ay = g0 + max(0, (g1 - g0 - (ah + gap + block_h)) // 2)
        top = ay + ah + gap
    elif L.get('center') or L['text_bottom'] - L['text_top'] - block_h > 60:   # also centre when lots of room is left
        # bias = share of the spare room placed above the text (0.5 = centred; lower sits closer to the art)
        top = top + int(max(0, L['text_bottom'] - L['text_top'] - block_h) * L.get('bias', 0.5))
    tag_y = top
    head_y = tag_y + th + gap_tag
    sub_y = head_y + len(hlines) * lh + gap_sub
    by_y = (sub_y + len(sublines) * sub_lh if sublines else head_y + len(hlines) * lh) + gap_by

    # clip box for the rise/sink animation must include descenders (g, j, p, q, y)
    band_h = fH.getbbox('Agjpqy')[3] + round(8 * k)
    words = []
    for li, ln in enumerate(hlines):
        x = X
        for w in ln.split(' '):
            words.append((text_img(w, fH, 'white', height=band_h), x, head_y + li * lh))
            x += text_len(w + ' ', fH)
    NW = len(words)
    subs = [text_img(s, fSub, sub_col) for s in sublines]
    # authors: either "authors": [{"name", "avatar"}, ...] or the single "author" / "avatar" pair
    authors = cfg.get('authors') or [{'name': cfg['author'], 'avatar': cfg.get('avatar')}]
    by1 = text_img(' AND '.join(a['name'] for a in authors), fBy, (235, 235, 235))
    by2 = text_img(cfg['date'], fBy, (200, 200, 200)) if cfg.get('date') else None   # date is optional

    avs = []                                   # overlapping circular headshots, left to right
    for a in authors:
        path = find_headshot(a, base)
        if not path: continue
        big = AV * 4
        ph = Image.open(path).convert('RGB')
        s = min(ph.size); ph = ph.crop(((ph.width - s) // 2, (ph.height - s) // 2, (ph.width + s) // 2, (ph.height + s) // 2)).resize((big, big), Image.LANCZOS)
        m = Image.new('L', (big, big), 0); ImageDraw.Draw(m).ellipse((0, 0, big - 1, big - 1), fill=255)
        av = Image.new('RGBA', (big, big), (0, 0, 0, 0)); av.paste(ph, (0, 0), m)
        ImageDraw.Draw(av).ellipse((2, 2, big - 3, big - 3), outline=(255, 255, 255, 255), width=7)
        avs.append(av.resize((AV, AV), Image.LANCZOS))
    step = round(AV * 0.72)
    av_w = AV + step * (len(avs) - 1) if avs else 0
    by_x = X + (av_w + round(14 * k) if avs else 0)

    logo = None
    if cfg.get('logo', True) and fmt in LOGOS:
        spec = LOGOS[fmt]
        src = Image.open(os.path.join(LOGO_DIR, LOGO_FILES[spec['kind']])).convert('RGBA')
        src = src.crop(src.getbbox())
        lw = round(src.width * spec['h'] / src.height)
        logo = fade(src.resize((lw, spec['h']), Image.LANCZOS), LOGO_OPACITY)
        if cfg.get('cta') and spec['at'] == 'row':
            # the CTA button takes the right end of the byline row: logo moves up to the tag row (horizontal lockup)
            src = Image.open(os.path.join(LOGO_DIR, LOGO_FILES['horizontal'])).convert('RGBA'); src = src.crop(src.getbbox())
            lh_ = round(th * 0.9); lw = round(src.width * lh_ / src.height)
            logo = fade(src.resize((lw, lh_), Image.LANCZOS), LOGO_OPACITY)
            logo_xy = (X + TW - lw, tag_y + (th - lh_) // 2)
        elif spec['at'] == 'row':
            logo_xy = (X + TW - lw, by_y + AV // 2 - spec['h'] // 2)
        else:
            logo_xy = (X, spec['y'] - spec['h'])

    # optional call-to-action button (promos): red box, white condensed caps, right end of the byline row
    cta = None
    if cfg.get('cta'):
        ctxt = text_img(cfg['cta'].upper(), font(700, 75, round(26 * k)), 'white')
        bw_, bh_ = ctxt.width + round(32 * k), AV
        cta = Image.new('RGBA', (bw_, bh_), RED + (255,))
        cta.alpha_composite(ctxt, ((bw_ - ctxt.width) // 2 + round(2 * k), (bh_ - ctxt.height) // 2 + round(3 * k)))
        cta_c = (X + TW - bw_ / 2, by_y + AV / 2)

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
        p = eoq(clamp((t - 0.25) / 0.7)); p0 = eic(clamp((t - 9.0 - E) / 0.45))
        if p > p0:
            if seam == 'bottom':
                d.rectangle((ax + int(aw * p0), ay + ah, ax + int(aw * p), ay + ah + sw), fill=RED)
            else:  # vertical line on the art's left edge
                d.rectangle((ax - sw, ay + int(ah * p0), ax - 1, ay + int(ah * p)), fill=RED)

        # tag box
        p = eoq(clamp((t - 0.35) / 0.45)) * (1 - eic(clamp((t - 8.85 - E) / 0.4)))
        if p > 0:
            d.rectangle((X, tag_y, X + int(tw * p), tag_y + th), fill=RED)
            q = eoc(clamp((t - 0.55) / 0.4))
            if q > 0:
                box = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
                box.alpha_composite(fade(tagtxt, q), (round(12 * k), int(round(4 * k) + (1 - q) * 20 * k)))
                fr.alpha_composite(box.crop((0, 0, max(1, int(tw * p)), th)), (X, tag_y))

        # headline: first in, last out (words rise from / sink into a mask)
        for j, (im, x, y) in enumerate(words):
            q = eoq(clamp((t + 1 / FPS - j * 0.04) / 0.5))   # first word is already moving on frame 0
            qo = eic(clamp((t - (P - 0.36 - (NW - 1) * 0.03) - (NW - 1 - j) * 0.03) / 0.28))
            if q <= 0 or qo >= 1: continue
            off = int((1 - q) * band_h + qo * band_h)
            band = Image.new('RGBA', (im.width, band_h), (0, 0, 0, 0)); band.alpha_composite(im, (0, off))
            fr.alpha_composite(band, (int(x), y - round(4 * k)))

        # subhead + byline
        items = [(s, (X, sub_y + n * sub_lh), 0.7 + n * 0.1, 8.65 + E) for n, s in enumerate(subs)]
        if by2 is not None:
            items += [(by1, (by_x, by_y + round(3 * k)), 1.05, 8.5 + E), (by2, (by_x, by_y + round(27 * k)), 1.15, 8.5 + E)]
        else:   # name only: centre it on the avatar row
            items += [(by1, (by_x, by_y + AV // 2 - round(13 * k)), 1.05, 8.5 + E)]
        for im, (px, py), st, so in items:
            q = eoc(clamp((t - st) / 0.6)); qo = clamp((t - so) / 0.3)
            if q > 0 and qo < 1:
                fr.alpha_composite(fade(im, q * (1 - qo)), (px, int(py + (1 - q) * 18 * k + qo * 10 * k)))
        for n, av in enumerate(avs):
            q = clamp((t - 0.9 - n * 0.12) / 0.5); qo = clamp((t - 8.45 - E + n * 0.06) / 0.3)
            if q > 0 and qo < 1:
                sz = max(1, int(AV * eob(q) * (1 - qo * qo)))
                ax0 = X + n * step
                fr.alpha_composite(fade(av.resize((sz, sz), Image.LANCZOS), min(1, q * 2.5)), (ax0 + (AV - sz) // 2, by_y + (AV - sz) // 2))

        if cta is not None:   # pops in after the byline, gentle pulse while holding, pops out with the byline
            q = clamp((t - 1.2) / 0.5); qo = clamp((t - 8.4 - E) / 0.3)
            if q > 0 and qo < 1:
                pulse = 1 + 0.035 * max(0.0, math.sin(TAU * (t - 1.7) / 2.5)) ** 8 if t > 1.7 else 1
                sc = eob(q) * (1 - qo * qo) * pulse
                im = cta.resize((max(1, int(cta.width * sc)), max(1, int(cta.height * sc))), Image.LANCZOS)
                fr.alpha_composite(fade(im, min(1, q * 2.5) * (1 - qo)), (int(cta_c[0] - im.width / 2), int(cta_c[1] - im.height / 2)))
        if logo is not None:   # always on screen
            fr.alpha_composite(logo, logo_xy)
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
