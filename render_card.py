#!/usr/bin/env python3
"""Render looping article cards (art loop + animated headline panel) in several aspect ratios.

Usage:  python3 render_card.py example/article.json [--formats 1x1,9x16]

The art loop is any seamless-looping 16:9 video (see README for how to make one).
The card loop length matches the art loop length.
"""
import json, math, os, re, random, subprocess, sys, argparse
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
    '4x5':  dict(size=(1080, 1350), art=(0, 0, 1080, 608),    text_x=64, text_w=952, text_top=650,  text_bottom=1350, head=98, center=True, bias=0.3,
                 link_zone=170),   # bottom of the 4:5 frame is where social platforms put the link: copy must not grow into it
    # 9:16 safe zone (Reels / TikTok / Stories): top 250px (profile header), bottom 420px (caption, CTA),
    # right 140px (like / comment / share buttons) are kept clear of text, logo and key art
    '9x16': dict(size=(1080, 1920), art=(0, 0, 1080, 608),    text_x=64, text_w=852, text_top=0,    text_bottom=0,    head=112, group=(250, 1500)),
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


def highlight_marks(lines, f, phrase, color, vtop=0.2, vbot=0.24):
    """Marker strokes for `phrase` across wrapped `lines`. Returns [(line index, marker RGBA, dark line RGBA, x start, x end)];
    images share the line's text origin (0,0)."""
    full = ' '.join(lines); i0 = full.find(phrase); i1 = i0 + len(phrase); pos = 0; out = []
    rng = random.Random(phrase)
    size = f.spec[2]; cap_top = f.getbbox('H')[1]; baseline = f.getbbox('H')[3]
    for n, ln in enumerate(lines):
        a0, a1 = max(i0, pos) - pos, min(i1, pos + len(ln)) - pos
        pos += len(ln) + 1
        if a1 <= a0: continue
        seg = ln[a0:a1].strip()
        a0 = a0 + (len(ln[a0:a1]) - len(ln[a0:a1].lstrip()))
        x0 = text_len(ln[:a0], f); x1 = x0 + text_len(seg, f)
        pad = 0.16 * size; lpad = 0.07 * size; top = cap_top - vtop * size; bot = baseline + vbot * size
        dark = text_img(ln, f, (12, 12, 12))
        Wm, Hm = int(x1 + pad * 2 + 8), int(max(dark.height, bot + 0.2 * size))
        S4 = 4; m = Image.new('L', (Wm * S4, Hm * S4), 0)
        jit = lambda: rng.uniform(-0.045, 0.045) * size
        steps = max(6, int((x1 - x0) / (0.6 * size)))
        topl = [(x0 - lpad + (x1 - x0 + lpad + pad) * j / steps, top + jit()) for j in range(steps + 1)]
        botl = [(x0 - lpad + (x1 - x0 + lpad + pad) * j / steps, bot + jit()) for j in range(steps, -1, -1)]
        left = [(x0 - lpad - 0.05 * size, bot - 0.15 * size), (x0 - lpad + 0.03 * size, top + 0.2 * size)]     # ragged chisel-tip ends
        right = [(x1 + pad + 0.07 * size, top + 0.25 * size), (x1 + pad - 0.03 * size, bot - 0.1 * size)]
        ImageDraw.Draw(m).polygon([(px * S4, py * S4) for px, py in left + topl + right + botl], fill=255)
        m = m.resize((Wm, Hm), Image.LANCZOS)
        streak = np.array([0.94 + 0.06 * rng.random() for _ in range(Hm)], np.float32)[:, None]   # faint ink streaks
        marker = Image.new('RGBA', (Wm, Hm), tuple(color) + (255,))
        marker.putalpha(Image.fromarray((np.array(m, np.float32) * streak).astype(np.uint8)))
        out.append((n, marker, dark, x0 - lpad - 0.1 * size, x1 + pad + 0.1 * size))
    return out


def draw_marks(fr, marks, p, alpha, origin_of_line):
    """Draw marker strokes wiped on to progress p (0-1), with the text under them in black."""
    for n, marker, dark, hx0, hx1 in marks:
        cut = int(hx0 + (hx1 - hx0) * p)
        mk = marker.copy(); mk.paste((0, 0, 0, 0), (max(0, cut), 0, mk.width, mk.height))
        mask = np.array(mk)[..., 3].astype(np.float32) / 255
        da = np.array(dark); hh, ww = min(da.shape[0], mask.shape[0]), min(da.shape[1], mask.shape[1])
        da = da[:hh, :ww].copy(); da[..., 3] = (da[..., 3] * mask[:hh, :ww]).astype(np.uint8)
        ox, oy = origin_of_line(n)
        fr.alpha_composite(fade(mk, alpha), (int(ox), int(oy))); fr.alpha_composite(fade(Image.fromarray(da), alpha), (int(ox), int(oy)))


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


def render(cfg, base, fmt, out_path, preview=False):
    L = LAYOUTS[fmt]
    CW, CH = L['size']
    ax, ay, aw, ah = L['art']
    k = L['head'] / 82.0                      # scale factor relative to the 1x1 design
    RED = hex2rgb(cfg.get('accent', '#E4002B'))
    X, TW = L['text_x'], L['text_w']

    fH = font(700, 75, L['head']); lh = int(L['head'] * 1.0)
    fTag = font(700, 75, round(30 * k)); fSub = font(600, 100, round(27 * k)); fBy = font(700, 90, round(18 * k))
    AV = round(50 * k)
    if not cfg.get('date'):   # name-only byline (article standard): bigger name and photo
        bs = cfg.get('byline_scale', 1.45)
        fBy = font(700, 90, round(18 * k * bs)); AV = round(50 * k * bs)

    # --- build text pieces & measure block height
    tagtxt = text_img(cfg.get('tag') or 'THE BULWARK', fTag, 'white'); tw = tagtxt.width + round(24 * k); th = round(44 * k)
    hlines = balanced_wrap(cfg['headline'], fH, TW)
    if len(hlines) == 1:   # short headline: scale it up (max 1.5x) while it still fits on one line
        hs = L['head']
        while hs < L['head'] * 1.5 and text_len(cfg['headline'], font(700, 75, hs + 2)) <= TW: hs += 2
        fH = font(700, 75, hs); lh = hs; hlines = [cfg['headline']]
    sublines = balanced_wrap(cfg['subhead'], fSub, TW) if cfg.get('subhead') else []
    sub_lh = round(36 * k)
    sub_col = (215, 215, 215)
    if cfg.get('subhead_prominent'):   # promos: the offer line is bigger, bolder and white
        ssc = cfg.get('subhead_scale', 1.0)   # extra size for short subheads
        fSub = font(700, 100, round(36 * k * ssc)); sub_lh = round(46 * k * ssc); sub_col = (255, 255, 255)
        sublines = balanced_wrap(cfg['subhead'], fSub, TW)
    # gap_tag = visible gap from the tag box to the top of the headline's capitals.
    # The font has built-in space above its capitals (bbox top of 'H'); subtract it, and add back the 4px the mask box is raised by.
    gap_tag = round(26 * k) - fH.getbbox('H')[1] + round(4 * k)
    gap_sub = round(24 * k); gap_by = round(24 * k)
    show_by = cfg.get('byline', True) is not False
    def measure():
        return th + gap_tag + len(hlines) * lh + (gap_sub + len(sublines) * sub_lh if sublines else 0) + (gap_by + AV if show_by else 0)
    block_h = measure()
    if not show_by:
        # no byline: give the room to the copy. The subhead switches to the bold white style and grows more (max 1.25x);
        # the headline grows only a little (max 1.1x), so the two stay readable without the headline overpowering it.
        if not cfg.get('subhead_prominent'):
            fSub = font(700, 100, round(36 * k)); sub_lh = round(46 * k); sub_col = (255, 255, 255)
        if L.get('group'): room = L['group'][1] - L['group'][0] - ah - round(44 * k)
        else: room = L['text_bottom'] - L['text_top'] - X - L.get('link_zone', 0)
        base_h, base_s, base_sl = fH.spec[2], fSub.spec[2], sub_lh
        sub_spec = fSub.spec
        best = None
        for step in range(0, 26):
            f_ = 1 + step * 0.01
            fH = font(700, 75, round(base_h * min(f_, 1.10))); lh = fH.spec[2]
            hlines = balanced_wrap(cfg['headline'], fH, TW)
            fSub = font(sub_spec[0], sub_spec[1], round(base_s * min(f_, 1.25))); sub_lh = round(base_sl * min(f_, 1.25))
            sublines = balanced_wrap(cfg['subhead'], fSub, TW) if cfg.get('subhead') else []
            gap_tag = round(26 * k) - fH.getbbox('H')[1] + round(4 * k)
            if measure() <= room: best = f_
            else: break
        f_ = best or 1.0
        fH = font(700, 75, round(base_h * min(f_, 1.10))); lh = fH.spec[2]; hlines = balanced_wrap(cfg['headline'], fH, TW)
        fSub = font(sub_spec[0], sub_spec[1], round(base_s * min(f_, 1.25))); sub_lh = round(base_sl * min(f_, 1.25))
        sublines = balanced_wrap(cfg['subhead'], fSub, TW) if cfg.get('subhead') else []
        gap_tag = round(26 * k) - fH.getbbox('H')[1] + round(4 * k)
        block_h = measure()
    # long copy: if the block won't fit above the bottom logo (16:9), shrink the headline until it does
    spec_ = LOGOS.get(fmt, {})
    if show_by and spec_.get('at') == 'bottom' and cfg.get('logo', True):
        limit = spec_['y'] - spec_['h'] - round(36 * k) - L['text_top']
        while block_h > limit and fH.spec[2] > L['head'] * 0.7:
            fH = font(700, 75, fH.spec[2] - 2); lh = fH.spec[2]
            hlines = balanced_wrap(cfg['headline'], fH, TW)
            gap_tag = round(26 * k) - fH.getbbox('H')[1] + round(4 * k)
            block_h = measure()
    top = L['text_top']
    if L.get('group'):   # centre art + text together inside a safe zone (vertical formats)
        g0, g1 = L['group']; gap = round(44 * k)
        ay = g0 + max(0, (g1 - g0 - (ah + gap + block_h)) // 2)
        top = ay + ah + gap
    elif L.get('center') or L['text_bottom'] - L.get('link_zone', 0) - L['text_top'] - block_h > 60:   # also centre when lots of room is left
        # bias = share of the spare room placed above the text (0.5 = centred; lower sits closer to the art)
        tb = L['text_bottom'] - L.get('link_zone', 0)   # never into the platform's link bar (4:5)
        if show_by and spec_.get('at') == 'bottom' and cfg.get('logo', True):   # never centre into the bottom logo
            tb = spec_['y'] - spec_['h'] - round(36 * k)
        top = top + int(max(0, tb - L['text_top'] - block_h) * L.get('bias', 0.5))
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

    # optional highlighter: a phrase in the headline (checked first) or subhead gets a marker stroke that swipes on;
    # the text under it turns black
    HL_COL = hex2rgb(cfg.get('highlight_color', '#FFF200'))   # neon yellow
    hl_head, hl_sub = [], []
    if cfg.get('highlight'):
        if cfg['highlight'] in ' '.join(hlines): hl_head = highlight_marks(hlines, fH, cfg['highlight'], HL_COL, vtop=0.09, vbot=0.12)   # tight headline leading
        elif sublines and cfg['highlight'] in ' '.join(sublines): hl_sub = highlight_marks(sublines, fSub, cfg['highlight'], HL_COL)
        else: print(f"  warning: highlight {cfg['highlight']!r} not found in headline or subhead", flush=True)
    # authors: either "authors": [{"name", "avatar"}, ...] or the single "author" / "avatar" pair
    authors = cfg.get('authors') or ([{'name': cfg['author'], 'avatar': cfg.get('avatar')}] if cfg.get('author') else [{'name': ''}])
    by1 = text_img(' AND '.join(a['name'] for a in authors), fBy, (235, 235, 235))
    by2 = text_img(cfg['date'], fBy, (200, 200, 200)) if cfg.get('date') else None   # date is optional

    avs = []                                   # overlapping circular headshots, left to right
    for a in (authors if show_by else []):
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
        if not show_by or (cfg.get('cta') and spec['at'] == 'row'):
            # the byline row is taken by the CTA, or gone: logo moves up to the tag row (horizontal lockup)
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

    if preview:   # layout preview: one still frame, everything on screen, art = the unanimated still
        still = Image.open(os.path.join(base, cfg['art_still'])).convert('RGB').resize((aw, ah), Image.LANCZOS)
        P = 10.0; N = 1
    else:
        art_path = os.path.join(base, cfg['art_loop'])
        P = probe_duration(art_path); N = int(round(P * FPS))
    E = P - 10.0   # timings below were designed on a 10s loop; outro is anchored to the loop end
    if cfg.get('outro') is False: E = 1e9   # text stays on screen through the end of the loop

    if not preview:
      dec = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-i', art_path, '-vf', f'scale={aw}:{ah}:flags=lanczos',
                            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
      enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{CW}x{CH}',
                            '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'slow',
                            '-movflags', '+faststart', out_path], stdin=subprocess.PIPE)
    seam = L.get('seam', 'bottom'); sw = max(4, round(5 * k))
    for i in range(N):
        if preview:
            t = 5.0; art = still
        else:
            t = i / FPS
            raw = dec.stdout.read(aw * ah * 3)
            if len(raw) < aw * ah * 3: break
            art = Image.frombytes('RGB', (aw, ah), raw)
        fr = Image.new('RGBA', (CW, CH), (0, 0, 0, 255))
        fr.paste(art, (ax, ay))
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
            qo = eic(clamp((t - (P - 0.36 - (NW - 1) * 0.03) - (NW - 1 - j) * 0.03) / 0.28)) if E < 1e8 else 0
            if q <= 0 or qo >= 1: continue
            off = int((1 - q) * band_h + qo * band_h)
            band = Image.new('RGBA', (im.width, band_h), (0, 0, 0, 0)); band.alpha_composite(im, (0, off))
            fr.alpha_composite(band, (int(x), y - round(4 * k)))
        if hl_head:   # swipes on once the headline is in; fades just before the headline sinks out
            hs = (NW - 1) * 0.04 + 0.45
            p = eoc(clamp((t - hs) / 0.5)); qo = clamp((t - (P - 0.62 - (NW - 1) * 0.03)) / 0.2) if E < 1e8 else 0
            if p > 0 and qo < 1:
                draw_marks(fr, hl_head, p, 1 - qo, lambda n: (X, head_y + n * lh - round(4 * k)))

        # subhead + byline
        items = [(s, (X, sub_y + n * sub_lh), 0.7 + n * 0.1, 8.65 + E) for n, s in enumerate(subs)]
        if not show_by:
            pass
        elif by2 is not None:
            items += [(by1, (by_x, by_y + round(3 * k)), 1.05, 8.5 + E), (by2, (by_x, by_y + round(27 * k)), 1.15, 8.5 + E)]
        else:   # name only: centre it on the avatar row
            cap = fBy.getbbox('H'); items += [(by1, (by_x, by_y + AV // 2 - (cap[1] + cap[3]) // 2), 1.05, 8.5 + E)]
        for im, (px, py), st, so in items:
            q = eoc(clamp((t - st) / 0.6)); qo = clamp((t - so) / 0.3)
            if q > 0 and qo < 1:
                fr.alpha_composite(fade(im, q * (1 - qo)), (px, int(py + (1 - q) * 18 * k + qo * 10 * k)))
        if hl_sub:   # swipes on once the subhead has landed
            p = eoc(clamp((t - 1.35) / 0.45)); qo = clamp((t - 8.65 - E) / 0.3)
            if p > 0 and qo < 1:
                draw_marks(fr, hl_sub, p, 1 - qo, lambda n: (X, sub_y + n * sub_lh + qo * 10 * k))
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
        if preview:
            out.save(out_path); return
        if i == int(5 * FPS):
            out.save(out_path.replace('.mp4', '_still.png'))
        enc.stdin.write(out.tobytes())
    enc.stdin.close(); enc.wait(); dec.stdout.close(); dec.wait()


# Card types. An article's "type" fills in these defaults; anything set in the article's own settings wins.
TYPES = {
    'article': {                 # standard article cards: image + headline + byline only
        'show_subhead': False,
        'show_date': False,
    },
    'appeal': {                  # Bulwark+ appeals / promos for paid media
        'tag': 'JOIN THE BULWARK',
        'byline': False,         # no author row; the copy takes the room
        'outro': False,          # text stays on screen through the end of the loop
        'highlight_color': '#FFF200',
        'show_subhead': True,
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--formats', help='comma list, e.g. 1x1,9x16 (default: from config)')
    ap.add_argument('--preview', action='store_true', help='layout check: one still per format from art_still, plus a side-by-side sheet; no video')
    a = ap.parse_args()
    base = os.path.dirname(os.path.abspath(a.config))
    cfg = json.load(open(a.config))
    cfg = {**TYPES.get(cfg.get('type', 'article'), {}), **cfg}
    # the subhead / date stay in the article's settings for reference; the type decides whether they show
    if not cfg.get('show_subhead', True): cfg.pop('subhead', None)
    if not cfg.get('show_date', True): cfg.pop('date', None)
    fmts = a.formats.split(',') if a.formats else cfg.get('formats', list(LAYOUTS))
    prefix = os.path.join(base, cfg.get('output_prefix', 'out/card'))
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    if a.preview:
        paths = []
        for f in fmts:
            out = f'{prefix}_{f}_preview.png'; render(cfg, base, f, out, preview=True); paths.append(out)
        ims = [Image.open(p) for p in paths]; Hs = 900
        ims = [im.resize((round(im.width * Hs / im.height), Hs), Image.LANCZOS) for im in ims]
        sheet = Image.new('RGB', (sum(im.width for im in ims) + 40 * (len(ims) + 1), Hs + 80), (60, 60, 60)); x = 40
        for im in ims: sheet.paste(im, (x, 40)); x += im.width + 40
        # platform UI overlays, drawn on the sheet only: red = covered by app UI, keep important things out
        UI = {'9x16': [(0, 0, 1080, 250, 'profile / header'), (0, 1500, 1080, 1920, 'caption / CTA'), (940, 250, 1080, 1500, 'buttons')],
              '4x5': [(0, 1180, 1080, 1350, 'link / CTA bar')]}
        x = 40
        for f, im in zip(fmts, ims):
            CW, CH = LAYOUTS[f]['size']; sc = Hs / CH
            ov = Image.new('RGBA', im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
            for x0, y0, x1, y1, lab in UI.get(f, []):
                d.rectangle((x0 * sc, y0 * sc, x1 * sc, y1 * sc), fill=(255, 40, 40, 70), outline=(255, 60, 60, 220), width=2)
                d.text((x0 * sc + 6, y0 * sc + 4), lab, fill=(255, 255, 255, 230), font=font(700, 90, 16))
            base_ = sheet.crop((x, 40, x + im.width, 40 + im.height)).convert('RGBA'); base_.alpha_composite(ov)
            sheet.paste(base_.convert('RGB'), (x, 40)); x += im.width + 40
        sheet.save(f'{prefix}_layout_preview.png'); print('preview', f'{prefix}_layout_preview.png', flush=True)
        return
    for f in fmts:
        out = f'{prefix}_{f}.mp4'
        print('rendering', out, flush=True)
        render(cfg, base, f, out)


if __name__ == '__main__':
    main()
