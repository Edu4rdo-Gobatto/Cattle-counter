# Gera os vídeos de borda da Fase 5: sem_gado.mp4 (30 frames pretos) e curto.mp4 (10 frames do vídeo real).
import sys, cv2, numpy as np
from pathlib import Path
src, out_dir = sys.argv[1], Path(sys.argv[2])
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
w = cv2.VideoWriter(str(out_dir / "sem_gado.mp4"), fourcc, 30, (640, 480))
for _ in range(30):
    w.write(np.zeros((480, 640, 3), np.uint8))
w.release()
cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
wd, ht = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = cv2.VideoWriter(str(out_dir / "curto.mp4"), fourcc, fps, (wd, ht))
n = 0
while n < 10:
    ok, f = cap.read()
    if not ok: break
    w.write(f); n += 1
w.release(); cap.release()
for name in ("sem_gado.mp4", "curto.mp4"):
    c = cv2.VideoCapture(str(out_dir / name))
    print(name, int(c.get(cv2.CAP_PROP_FRAME_COUNT)), "frames", int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    c.release()
