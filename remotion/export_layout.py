#!/usr/bin/env python3
"""Export the template's layout (render_card.py) for one article as JSON for the Remotion cards.

Usage:  python3 remotion/export_layout.py articles/<slug>/article.json  -> remotion/public/<slug>/layout.json
Positions, wraps and font sizes come from render_card.py, so the Remotion cards match the template exactly."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import render_card as rc

cfg_path = sys.argv[1]; base = os.path.dirname(os.path.abspath(cfg_path)); slug = os.path.basename(base)
cfg = json.load(open(cfg_path)); cfg = {**rc.TYPES.get(cfg.get('type', 'article'), {}), **cfg}
if not cfg.get('show_subhead', True): cfg.pop('subhead', None)
if not cfg.get('show_date', True): cfg.pop('date', None)
out = {}
for f in cfg.get('formats', list(rc.LAYOUTS)):
    d = {}; rc.render(cfg, base, f, None, preview=True, export=d); out[f] = d
dst = os.path.join(HERE, 'public', slug); os.makedirs(dst, exist_ok=True)
json.dump(out, open(os.path.join(dst, 'layout.json'), 'w'), indent=1)
print(os.path.join(dst, 'layout.json'))
