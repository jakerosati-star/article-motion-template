"""Miller Daddy art: a slow push-in on the still and back out, so the 10s loop is seamless (frame 0 = the art).
Sampled from the full-size 3000px still for a sharp image. Run from art_build/. Writes ../art_loop.mp4"""
import math, subprocess
import numpy as np, cv2
from PIL import Image

W, H = 1920, 1080; P = 10.0; FPS = 30; N = int(P * FPS); TAU = 2 * math.pi
src = np.array(Image.open('../art_still.jpg').convert('RGB'))
s = W / src.shape[1]                       # full-size still -> 1920
CX, CY = 0.5 * W, 0.45 * H                 # push toward the two faces
ZOOM = 0.06                                # max push-in: 6%

enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', '-preset', 'slow', '-movflags', '+faststart', '../art_loop.mp4'],
                       stdin=subprocess.PIPE)
for i in range(N):
    z = 1 + ZOOM * (0.5 - 0.5 * math.cos(TAU * i / N))   # eases in, peaks at 5s, eases back out
    M = np.float32([[s * z, 0, (1 - z) * CX], [0, s * z, (1 - z) * CY]])
    f = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_AREA if z * s < 1 else cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    enc.stdin.write(f.tobytes())
enc.stdin.close(); enc.wait()
