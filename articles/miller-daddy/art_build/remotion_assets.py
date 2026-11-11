"""Art layers for the Remotion version of the Miller Daddy card (remotion/src/MillerArt.tsx animates them):
art.jpg (the still), sleeve.png (his extra jacket sleeve, rest pose) and hand.png (his hand, palm down, rest pose
on Trump's head), plus arm.json with where each layer sits and the shoulder / wrist pivots, all in 1920x1080 art px."""
import json, os
import numpy as np
from PIL import Image
import pat_proto as pp

OUT = '../../../remotion/public/miller-daddy'; os.makedirs(OUT, exist_ok=True)
WX, WY = 1010, 430; SHOULDER = (1340, 560)
Image.fromarray(pp.a.astype(np.uint8)).save(f'{OUT}/art.jpg', quality=93)

def save(layer, name):
    al = layer[..., 3]; ys, xs = np.where(al > 1)
    x0, y0, x1, y1 = xs.min() - 2, ys.min() - 2, xs.max() + 3, ys.max() + 3
    rgba = np.clip(layer[y0:y1, x0:x1], 0, 255).astype(np.uint8)
    Image.fromarray(rgba, 'RGBA').save(f'{OUT}/{name}.png')
    return dict(x=int(x0), y=int(y0), w=int(x1 - x0), h=int(y1 - y0))

sleeve = save(pp.sleeve_layer(WX + 10, WY + 8, SHOULDER[0], SHOULDER[1], 58, 90), 'sleeve')
hand_l, _ = pp.hand_layer(78, 0.8, WX, WY)
hand = save(hand_l, 'hand')
json.dump(dict(sleeve=sleeve, hand=hand, wrist=[WX, WY], shoulder=list(SHOULDER)), open(f'{OUT}/arm.json', 'w'), indent=1)
print(sleeve, hand)
