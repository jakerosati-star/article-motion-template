#!/usr/bin/env python3
"""Clean cut-outs of people (and other subjects) from a still, for the art-loop scripts.

The default is the AI person-detection model (rembg's u2net_human_seg). It gave the cleanest cut-out of everything
tried on these cards (Miller 2028: head, ears, raised hand and shoulders), with soft natural edges, and it only ever
picks people, so props, text and backgrounds never leak in. For non-people (animals, objects, cartoon cut-outs)
use --model isnet-general-use. Objects with straight edges (podiums, signs, frames) are better traced as a polygon.

Usage:
  python3 cutout.py <image> <out_mask.png> [--model u2net_human_seg] [--keep largest|all] [--box x0,y0,x1,y1]
    --keep largest  keep only the biggest person (default); all: every person found
    --box           only keep what overlaps this region (e.g. one person out of several)
The output is a soft (anti-aliased) 8-bit alpha mask the size of the image. In a script:
  from cutout import person_matte; alpha = person_matte(pil_image)    # float 0-1
"""
import argparse, sys
import numpy as np, cv2
from PIL import Image

def person_matte(im, model='u2net_human_seg', keep='largest', box=None):
    from rembg import remove, new_session
    m = np.array(remove(im.convert('RGB'), session=new_session(model), only_mask=True)).astype(np.float32) / 255
    b = (m > 0.5).astype(np.uint8)
    if box is not None:
        x0, y0, x1, y1 = box; roi = np.zeros_like(b); roi[y0:y1, x0:x1] = 1; b &= roi
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    if n < 2: return np.zeros_like(m)
    ids = [1 + int(np.argmax(st[1:, 4]))] if keep == 'largest' else [i for i in range(1, n) if st[i, 4] > 0.002 * b.size]
    region = cv2.dilate(np.isin(lab, ids).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    return m * region                                   # soft edges kept, strays dropped

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('image'); ap.add_argument('out')
    ap.add_argument('--model', default='u2net_human_seg'); ap.add_argument('--keep', default='largest', choices=['largest', 'all'])
    ap.add_argument('--box'); a = ap.parse_args()
    box = tuple(int(v) for v in a.box.split(',')) if a.box else None
    alpha = person_matte(Image.open(a.image), a.model, a.keep, box)
    Image.fromarray((alpha * 255).astype(np.uint8)).save(a.out); print(a.out)
