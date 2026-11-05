# Acha o início da vinheta escura e cruza com os IDs confirmados do relatório.
import sys, csv, cv2
video, ids_csv = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(video); i = 0; start = None
while True:
    ok, f = cap.read()
    if not ok: break
    i += 1  # índice 1-based, igual ao counter._frame_index
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    if start is None and g[100:900].mean() < 40: start = i
cap.release()
print("frames", i, "inicio vinheta (1-based)", start)
rows = list(csv.DictReader(open(ids_csv, encoding="utf-8")))
conf = [r for r in rows if r["confirmed"] == "1"]
viv = [r for r in conf if int(r["last_frame"]) >= start]
print("confirmados", len(conf), "| confirmados vivos na vinheta", len(viv))
for r in viv: print(r)
