import numpy as np, cv2, math
from PIL import Image
from scipy import ndimage
a=np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32); H,W=a.shape[:2]
hs=np.array(Image.open('hand_src.png')).astype(np.float32)
al=hs[...,3]/255; al=ndimage.binary_opening(al>0.5,iterations=3); lab,n=ndimage.label(al); al=lab==1+int(np.argmax(ndimage.sum(np.ones_like(lab),lab,range(1,n+1))))
hs[...,3]=ndimage.gaussian_filter(al.astype(np.float32),0.8)*255
h,w=hs.shape[:2]
# sleeve: a strip of Miller's jacket sleeve (his right forearm), tiled along a path from the wrist up-right
sl=a[620:900,1300:1380].copy()   # jacket sleeve texture
def hand_layer(ang, sc, wx, wy):
    # pivot = wrist (bottom-centre of the cuff in the source)
    piv=(w*0.72, h*0.97)
    M=cv2.getRotationMatrix2D(piv, ang, sc); M[0,2]+=wx-piv[0]; M[1,2]+=wy-piv[1]
    return cv2.warpAffine(hs,M,(W,H),flags=cv2.INTER_LINEAR,borderValue=0), M
def sleeve_layer(wx, wy, ex, ey, w0, w1):
    """tapered, shaded sleeve from the wrist (w0 wide) to the shoulder (w1 wide), Miller's jacket texture"""
    L = int(math.hypot(ex - wx, ey - wy)); Wd = max(w0, w1)
    tex = cv2.resize(sl, (Wd, L))
    u = (np.arange(Wd) - Wd / 2) / (Wd / 2)
    shade = 0.55 + 0.45 * np.cos(np.clip(u, -1, 1) * 1.2)               # round: lit centre, darker edges
    rgba = np.dstack([tex * shade[None, :, None], np.zeros((L, Wd), np.float32)])
    half = (w0 + (w1 - w0) * np.arange(L) / L)[:, None] / 2
    rgba[..., 3] = 255 * np.clip(half - np.abs(np.arange(Wd)[None, :] - Wd / 2) + 0.5, 0, 1)
    rgba[..., 3] *= np.clip(np.arange(L)[:, None] / 8, 0, 1)            # soft at the wrist end
    src = np.float32([[Wd / 2, 0], [Wd / 2, L], [0, 0]])
    dx, dy = (ex - wx) / L, (ey - wy) / L; nx, ny = -dy, dx
    dst = np.float32([[wx, wy], [ex, ey], [wx - nx * Wd / 2, wy - ny * Wd / 2]])
    return cv2.warpAffine(rgba, cv2.getAffineTransform(src, dst), (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
def over(f,l): al=l[...,3:4]/255; return f*(1-al)+l[...,:3]*al
if __name__=='__main__':
    f=a.copy()
    WX,WY=1010,430
    f=over(f,sleeve_layer(WX+10,WY+8,1340,560,64,96))
    lay,_=hand_layer(78,0.8,WX,WY); f=over(f,lay)
    Image.fromarray(np.clip(f,0,255).astype(np.uint8)).crop((560,150,1500,800)).save('/tmp/claude-0/-home-user-article-motion-template/fbb3ffda-e686-5ff7-bf3e-cd2fa0bae43e/scratchpad/pat.png')
