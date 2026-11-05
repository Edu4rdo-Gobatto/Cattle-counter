# Recorta faixas horizontais do frame bruto (sem caixas) e amplia 2x para contagem manual.
import sys, cv2
from pathlib import Path
video, out_dir = sys.argv[1], Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
bands = [(230, 340), (330, 440), (430, 560), (550, 830)]
for f in (15, 98, 149):
    cap = cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, frame = cap.read(); cap.release()
    assert ok, f
    cv2.imwrite(str(out_dir / f"raw_f{f:05d}.png"), frame)
    for i, (y0, y1) in enumerate(bands, 1):
        crop = frame[y0:y1]
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # Régua: marcas a cada 50 px (coordenada original) para auditar a contagem
        for x in range(0, frame.shape[1], 50):
            cv2.line(crop, (2 * x, 0), (2 * x, 12), (0, 0, 255), 2)
            cv2.putText(crop, str(x), (2 * x + 3, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(str(out_dir / f"f{f:05d}_faixa{i}_y{y0}-{y1}.png"), crop)
print("ok")
