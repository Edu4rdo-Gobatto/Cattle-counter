# Extrai um frame do vídeo anotado na fração pedida (ex.: 0.10) e salva como PNG.
import sys, cv2
video, frac, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
cap = cv2.VideoCapture(video)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
idx = min(n - 1, int(n * frac))
cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
ok, frame = cap.read(); cap.release()
assert ok, f"falha lendo frame {idx} de {video}"
cv2.imwrite(out, frame)
print(f"{video}: frame {idx}/{n} -> {out}")
