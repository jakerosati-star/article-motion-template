#!/usr/bin/env python3
"""Turn already-animated article art (GIF / animated WebP) into a 1920x1080, 30fps art loop.

Usage:  python3 gif_to_art_loop.py articles/<slug>/<art>.gif articles/<slug>/art_loop.mp4

Frame timings are kept exactly; the animation is repeated a whole number of times to get as close to 10s as possible.
"""
import subprocess, sys
from PIL import Image, ImageSequence

src, dst = sys.argv[1], sys.argv[2]
im = Image.open(src)
frames = [(f.convert('RGB').resize((1920, 1080), Image.LANCZOS).tobytes(), f.info.get('duration') or 100) for f in ImageSequence.Iterator(im)]
cycle = sum(d for _, d in frames) / 1000
reps = max(1, round(10 / cycle))
print(f'{len(frames)} frames, {cycle:.2f}s per cycle, repeated {reps}x = {cycle * reps:.2f}s')
enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1920x1080', '-r', '30', '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', '-preset', 'slow', '-movflags', '+faststart', dst],
                       stdin=subprocess.PIPE)
for _ in range(reps):
    for b, d in frames:
        for _ in range(round(d * 30 / 1000)): enc.stdin.write(b)
enc.stdin.close(); enc.wait()
