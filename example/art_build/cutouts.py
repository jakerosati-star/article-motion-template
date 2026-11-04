import os
import numpy as np, cv2
from PIL import Image
from rembg import remove, new_session
src = Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','art_still.webp')).convert('RGB')
boxes = {'blimp':(935,45,1265,170),'cat':(85,410,270,655),'st_tl':(85,0,225,138),'st_g':(740,465,825,555),'st_r':(1218,345,1272,435)}
s = new_session('isnet-general-use')
W,H = src.size
for k,(x0,y0,x1,y1) in boxes.items():
    crop = src.crop((x0,y0,x1,y1))
    out = remove(crop, session=s)
    out.save(f'{k}.png')
    # preview on checker
    bg = Image.new('RGB', out.size, (255,0,255)); bg.paste(out, mask=out.split()[3]); bg.save(f'prev_{k}.png')
