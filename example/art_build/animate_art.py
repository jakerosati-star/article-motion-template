import os
import numpy as np, cv2, math, random, subprocess
from PIL import Image, ImageDraw, ImageFilter
SRC=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','art_still.webp')
src=Image.open(SRC).convert('RGB'); W,H=src.size
boxes={'blimp':(935,45,1265,170),'cat':(85,410,270,655),'st_tl':(85,0,225,138),'st_g':(740,465,825,555),'st_r':(1218,345,1272,435)}
layers={}; hole=np.zeros((H,W),np.uint8)
for k,(x0,y0,x1,y1) in boxes.items():
    a=np.array(Image.open(f'{k}.png'))[:,:,3]
    full=np.zeros((H,W),np.uint8); full[y0:y1,x0:x1]=a
    if k=='cat':
        m=Image.new('L',(W,H),0)
        ImageDraw.Draw(m).polygon([(150,420),(187,416),(224,420),(222,500),(243,502),(232,515),(205,528),(175,530),(150,522),(133,506),(152,500)],fill=255)
        full=np.maximum(full,np.array(m.filter(ImageFilter.GaussianBlur(0.8))))
    full[full<25]=0
    layers[k]=full
    hole=np.maximum(hole,(full>20).astype(np.uint8)*255)
hole=cv2.dilate(hole,np.ones((9,9),np.uint8))
bgarr=cv2.inpaint(np.array(src)[:,:,::-1],hole,9,cv2.INPAINT_TELEA)[:,:,::-1]
# add a little grain into filled areas so they don't look smeared
noise=np.random.default_rng(1).normal(0,5,bgarr.shape)
bgarr=np.where(hole[...,None]>0,np.clip(bgarr+noise,0,255),bgarr).astype(np.uint8)
cm=np.load('catmask.npy'); sx=np.clip(2*262-np.arange(W),0,W-1)
donor=np.ascontiguousarray(np.array(src)[:,sx][:,:,::-1])
ys_,xs_=np.nonzero(cm); cc=(int((xs_.min()+xs_.max())/2),int((ys_.min()+ys_.max())/2))
bgarr=cv2.seamlessClone(donor,np.ascontiguousarray(bgarr[:,:,::-1]),cm,cc,cv2.NORMAL_CLONE)[:,:,::-1].copy()
Image.fromarray(bgarr).save('bg_clean.png')
srcarr=np.array(src)
sprites={}
for k,(x0,y0,x1,y1) in boxes.items():
    ys,xs=np.nonzero(layers[k]); bx0,by0,bx1,by1=xs.min(),ys.min(),xs.max()+1,ys.max()+1
    rgba=np.dstack([srcarr[by0:by1,bx0:bx1],layers[k][by0:by1,bx0:bx1]])
    sprites[k]=(Image.fromarray(rgba,'RGBA'),bx0,by0)

# laser line & glow sprites
laser_p0=(655,185); laser_p1=(1272,520)
def glow_line(width,blur):
    m=Image.new('L',(W,H),0); ImageDraw.Draw(m).line([laser_p0,laser_p1],fill=255,width=width)
    return np.array(m.filter(ImageFilter.GaussianBlur(blur)),np.float32)/255
laser_glow=glow_line(18,14); laser_core=glow_line(5,3)
eye=Image.new('L',(W,H),0); ImageDraw.Draw(eye).ellipse((585,115,680,150),fill=255)
eye_glow=np.array(eye.filter(ImageFilter.GaussianBlur(12)),np.float32)/255
# light ray shimmer: rotating radial pattern centered at ray source
cx,cy=1130,95
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
ang=np.arctan2(yy-cy,xx-cx); dist=np.hypot(xx-cx,yy-cy)
falloff=np.exp(-dist/420)*(dist>40)
skymask=np.array(Image.fromarray(255-cv2.dilate((np.abs(srcarr.astype(int)-[30,110,230]).sum(2)>170).astype(np.uint8)*255,np.ones((3,3)))).filter(ImageFilter.GaussianBlur(3)),np.float32)/255

P=10.0; FPS=30; N=int(P*FPS); TAU=2*math.pi
def clamp(x): return max(0.0,min(1.0,x))
def eoc(t): return 1-(1-t)**3
def eic(t): return t**3
def eob(t,s=1.9): t-=1; return 1+(s+1)*t**3+s*t**2
def eib(t,s=1.9): return (s+1)*t**3-s*t**2
def ease_io(t): return t*t*(3-2*t)
def wv(k,t,ph=0): return math.sin(TAU*k*t/P+ph)   # periodic over P

# periodic confetti
rng=random.Random(7); Ly=H+160; conf=[]
for i in range(48):
    conf.append(dict(x=rng.uniform(-30,W+30),y0=rng.uniform(0,Ly),n=rng.choice([1,1,1,2]),sway=rng.uniform(10,35),
        m=rng.choice([1,2,3]),ph=rng.uniform(0,TAU),rot=rng.uniform(0,360),r=rng.choice([-5,-4,-3,-2,2,3,4,5]),
        w=rng.uniform(10,22),h=rng.uniform(5,9),s=rng.randint(4,10),col=rng.choice([(20,40,140),(30,70,200),(10,25,95),(60,120,235)])))
def draw_confetti(img,t):
    for c in conf:
        y=(c['y0']+Ly*c['n']*t/P)%Ly-80
        x=c['x']+c['sway']*wv(c['m'],t,c['ph'])
        flip=abs(math.cos(TAU*c['s']*t/P+c['ph']))
        w,h=c['w'],max(1.5,c['h']*flip)
        piece=Image.new('RGBA',(int(w)+4,int(h)+4),(0,0,0,0))
        shade=0.65+0.35*flip; col=tuple(int(v*shade+30*(1-flip)) for v in c['col'])
        ImageDraw.Draw(piece).rectangle((2,2,2+w,2+h),fill=col+(235,))
        piece=piece.rotate(c['rot']+360*c['r']*t/P,expand=True,resample=Image.BICUBIC)
        img.alpha_composite(piece,(int(x-piece.width/2),int(y-piece.height/2)))

def xform(spr,scale,rot,alpha):
    s=spr
    if scale!=1.0: s=s.resize((max(1,int(spr.width*scale)),max(1,int(spr.height*scale))),Image.BICUBIC)
    if rot: s=s.rotate(rot,expand=True,resample=Image.BICUBIC)
    if alpha<1:
        a=np.array(s); a[:,:,3]=(a[:,:,3]*alpha).astype(np.uint8); s=Image.fromarray(a)
    return s
def place(frame,k,dx=0,dy=0,scale=1.0,rot=0.0,alpha=1.0,anchor='center'):
    spr,x0,y0=sprites[k]
    if alpha<=0 or scale<=0.02: return
    s=xform(spr,scale,rot,alpha)
    ax=x0+spr.width/2
    if anchor=='bottom': ay=y0+spr.height; px=ax-s.width/2; py=ay-spr.height*scale/2-s.height/2
    else: ay=y0+spr.height/2; px=ax-s.width/2; py=ay-s.height/2
    frame.alpha_composite(s,(int(round(px+dx)),int(round(py+dy))))
def place_at(frame,k,cx,cy,scale,rot,alpha=1.0):
    if scale<=0.02 or alpha<=0: return
    s=xform(sprites[k][0],scale,rot,alpha)
    frame.alpha_composite(s,(int(cx-s.width/2),int(cy-s.height/2)))

def in_out(t,tin,tout,din=0.45,dout=0.35):
    """returns (phase, u): 'in' with eased progress, 'hold', 'out', or None"""
    if t<tin: return None,0
    if t<tin+din: return 'in',(t-tin)/din
    if t<tout: return 'hold',1
    if t<tout+dout: return 'out',(t-tout)/dout
    return None,0
def sticker(frame,k,cx,cy,sc,r0,tin,tout):
    ph,u=in_out(t,tin,tout)
    if ph=='in': place_at(frame,k,cx,cy,sc*(1+1.6*(1-eoc(u))),r0+(-30)*(1-eob(u,2.2)),clamp(u*3))
    elif ph=='hold': place_at(frame,k,cx,cy,sc,r0)
    elif ph=='out': place_at(frame,k,cx,cy,sc*(1-eib(u,1.6)),r0+90*u)
# tossed stickers: (sprite, t0, t1, start(x,y), end(x,y), arc height, spin deg, scale)
FLY=[('st_g',3.7,5.3,(170,H+70),(760,H+70),520,560,1.5),
     ('st_g',5.4,6.9,(W+70,330),(-70,170),140,-720,1.15),
     ('st_g',6.6,8.2,(1180,H+70),(860,H+70),470,-480,1.7),
     ('st_g',7.3,8.6,(-70,90),(560,-80),-60,420,1.0)]
POPS=[('st_g',470,105,1.35,18,4.2,6.1),('st_g',1010,600,1.55,-22,5.0,7.3),('st_tl',330,300,0.75,24,6.2,7.9),('st_g',560,560,1.2,8,7.0,8.15)]

bgf=bgarr.astype(np.float32)
ff=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
    '-c:v','libx264','-pix_fmt','yuv420p','-crf','17','-preset','slow','-movflags','+faststart','../art_loop.mp4'],stdin=subprocess.PIPE)
for i in range(N):
    t=i/FPS
    f=bgf.copy()
    rays=(0.5+0.5*np.sin(ang*26+TAU*t/P))**6*falloff*skymask*(0.55+0.15*wv(3,t))
    f+= rays[...,None]*np.array([90,150,200],np.float32)*0.6
    p=0.55+0.45*wv(12,t)*wv(4,t,1)
    f+= laser_glow[...,None]*np.array([60,180,255],np.float32)*(0.35+0.4*p)
    f+= laser_core[...,None]*np.array([150,240,255],np.float32)*(0.25*p)
    flick=0.55+0.45*abs(wv(14,t)*wv(6,t))
    f+= eye_glow[...,None]*np.array([255,40,30],np.float32)*(0.3*flick)
    frame=Image.fromarray(np.clip(f,0,255).astype(np.uint8)).convert('RGBA')
    # blimp: in from right, out to the right
    tb=eoc(clamp((t-0.2)/2.8))-eic(clamp((t-8.8)/1.1))
    place(frame,'blimp',dx=380*(1-tb),dy=4*wv(2,t),rot=1.2*wv(2,t)*tb)
    # cat pops up, ducks down
    if 1.1<=t<9.4:
        if t<2.0: tc=1.1; sc=eob((t-1.1)/0.9,1.6); r=4*(1-(t-1.1)/0.9)*math.sin((t-1.1)/0.9*12)
        elif t<8.9: sc=1; r=0
        else: sc=1-eib((t-8.9)/0.5,1.4); r=0
        place(frame,'cat',scale=max(0.02,sc),anchor='bottom',rot=r)
    # original stickers: slap in, pop off at the end
    for k,st,r0,to in (('st_tl',2.0,-35,8.75),('st_g',2.5,30,8.6),('st_r',3.0,-25,8.45)):
        ph,u=in_out(t,st,to)
        if ph=='in': e=eob(u,2.2); place(frame,k,scale=1+1.6*(1-eoc(u)),rot=r0*(1-e),alpha=clamp(u*3))
        elif ph=='hold': place(frame,k)
        elif ph=='out': place(frame,k,scale=1-eib(u,1.6),rot=-60*u)
    # extra pop-in stickers
    for k,cx,cy,sc,r0,tin,tout in POPS: sticker(frame,k,cx,cy,sc,r0,tin,tout)
    # tossed stickers
    for k,t0,t1,(xs,ys),(xe,ye),hh,spin,sc in FLY:
        if t0<=t<=t1:
            u=(t-t0)/(t1-t0)
            place_at(frame,k,xs+(xe-xs)*u, ys+(ye-ys)*u-4*hh*u*(1-u), sc, spin*u)
    draw_confetti(frame,t)
    out=frame.convert('RGB')
    z=1.0+0.045*(0.5-0.5*math.cos(TAU*t/P))
    cxm=W/2+8*wv(1,t); cym=H/2+5*wv(1,t,0.7)-5*math.sin(0.7)
    M=np.array([[z,0,(1-z)*cxm],[0,z,(1-z)*cym]],np.float32)
    o=cv2.warpAffine(np.array(out),M,(W,H),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT)

    ff.stdin.write(o.tobytes())
ff.stdin.close(); ff.wait()
