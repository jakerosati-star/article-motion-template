"""Turn the article's animated GIF into a 10s, 30fps, 1920x1080 art loop (whole GIF cycles, timings kept)."""
import subprocess, os
from PIL import Image, ImageSequence
os.chdir(os.path.dirname(os.path.abspath(__file__)))
im = Image.open('art_original.gif')
frames = [(f.convert('RGB').resize((1920, 1080), Image.LANCZOS).tobytes(), f.info['duration']) for f in ImageSequence.Iterator(im)]
cycle = sum(d for _, d in frames) / 1000
reps = max(1, round(10 / cycle))
enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1920x1080', '-r', '30', '-i', '-',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', '-preset', 'slow', '-movflags', '+faststart', 'art_loop.mp4'],
                       stdin=subprocess.PIPE)
for _ in range(reps):
    for b, d in frames:
        for _ in range(round(d * 30 / 1000)): enc.stdin.write(b)
enc.stdin.close(); enc.wait()
