"""Rebuild the parts of the bill stack hidden behind Vought and the dome, from a reference image of the same note
(ref_100_series2006A.jpg: Wikimedia Commons 'Usdollar100front.jpg', US government work, public domain).
Each bill's position is found by SIFT matching + RANSAC homography, one bill at a time."""
import numpy as np, cv2
from PIL import Image
from scipy import ndimage
a = np.array(Image.open('still_1920.png').convert('RGB')).astype(np.float32); H, W = a.shape[:2]
P = dict(np.load('pieces.npz'))
ref = cv2.imread('ref_100_series2006A.jpg', cv2.IMREAD_GRAYSCALE)
sc = 1600 / ref.shape[1]; ref = cv2.resize(ref, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA); rh, rw = ref.shape
gray = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY)
occl = ((P['vought'] > 0.3) | (P['dome'] > 0.3))
vis = (P['bills'] > 0.5) & ~occl
cv2.setRNGSeed(0)
sift = cv2.SIFT_create(nfeatures=20000)
kr, dr = sift.detectAndCompute(ref, None)
ka, da = sift.detectAndCompute(gray, (vis * 255).astype(np.uint8))
m = cv2.BFMatcher().knnMatch(dr, da, k=2)
good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
src = np.float32([kr[g.queryIdx].pt for g in good]); dst = np.float32([ka[g.trainIdx].pt for g in good])
alive = np.ones(len(good), bool); bills = []
corners = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]]).reshape(-1, 1, 2)
for _ in range(6):
    if alive.sum() < 12: break
    Hm, inl = cv2.findHomography(src[alive], dst[alive], cv2.RANSAC, 4.0)
    if Hm is None or inl.sum() < 15: break
    q = cv2.perspectiveTransform(corners, Hm).reshape(-1, 2)
    quad = np.zeros((H, W), np.uint8); cv2.fillPoly(quad, [q.astype(np.int32)], 1)
    idx = np.where(alive)[0]
    inside = quad[np.clip(dst[idx, 1].astype(int), 0, H - 1), np.clip(dst[idx, 0].astype(int), 0, W - 1)] > 0
    alive[idx[inside]] = False
    bills.append((Hm, q, int(inl.sum())))
    print('bill', len(bills), 'inliers', int(inl.sum()), 'corners', q.round().astype(int).tolist())
# tone-match the reference to the art's bills (art bills are high-contrast greyscale)
qs = np.linspace(0, 100, 101)
src_q = np.percentile(ref, qs); dst_q = np.percentile(gray[vis], qs)
lut = np.interp(np.arange(256), src_q, dst_q).astype(np.float32)
ref_t = lut[ref]
# lower bills first (bottom-most first); the top bill is drawn last as one BENT bill: the photo's top note curves,
# so its left part fits bills[0] and the sliver right of Vought fits H_sliver. Blend the two mappings across x.
order = sorted(bills[1:], key=lambda b: -b[1][:, 1].mean())
recon = np.zeros((H, W), np.float32); cover = np.zeros((H, W), np.float32)
for Hm, q, _ in order:
    w = cv2.warpPerspective(ref_t, Hm, (W, H), flags=cv2.INTER_LINEAR)
    mk = cv2.warpPerspective(np.ones_like(ref_t), Hm, (W, H), flags=cv2.INTER_LINEAR)
    recon = recon * (1 - mk) + w * mk; cover = np.maximum(cover, mk)
Hs = np.load('H_sliver.npy')
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
def inv_map(Hm):
    Hi = np.linalg.inv(Hm); d = Hi[2, 0] * xx + Hi[2, 1] * yy + Hi[2, 2]
    return (Hi[0, 0] * xx + Hi[0, 1] * yy + Hi[0, 2]) / d, (Hi[1, 0] * xx + Hi[1, 1] * yy + Hi[1, 2]) / d
u1, v1 = inv_map(bills[0][0]); u2, v2 = inv_map(Hs)
wgt = np.clip((xx - 880) / 240, 0, 1); wgt = wgt * wgt * (3 - 2 * wgt)
mu = ((1 - wgt) * u1 + wgt * u2).astype(np.float32); mv = ((1 - wgt) * v1 + wgt * v2).astype(np.float32)
top = cv2.remap(ref_t, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
tm = cv2.remap(np.ones_like(ref_t), mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
recon = recon * (1 - tm) + top * tm; cover = np.maximum(cover, tm)
recon_rgb = np.repeat(recon[..., None], 3, 2)
# new bill layer: original pixels where visible, reconstruction where Vought / the dome covered them
hidden = ndimage.binary_dilation(P['vought'] > 0.3, iterations=3) & (cover > 0.5) & ~(P['dome'] > 0.3)   # only what Vought covered
alpha = np.maximum(P['bills'], hidden.astype(np.float32))
blend = ndimage.gaussian_filter(hidden.astype(np.float32), 1.5)
rgb = a * (1 - blend[..., None]) + recon_rgb * blend[..., None]
np.save('bills_rgb.npy', rgb.astype(np.uint8)); P['bills_full'] = alpha.astype(np.float32); np.savez_compressed('pieces.npz', **P)
v = rgb * alpha[..., None] + np.array([0, 200, 80]) * (1 - alpha[..., None])
Image.fromarray(v.astype(np.uint8)).crop((300, 60, 1450, 1080)).resize((805, 714)).save('/tmp/claude-0/-home-user-article-motion-template/fbb3ffda-e686-5ff7-bf3e-cd2fa0bae43e/scratchpad/bills_full.png')
