# Modelos (pesos)

Os arquivos `*.pt` são grandes e estão no `.gitignore`, por isso **não vão para o
repositório**. Baixe o modelo adotado e salve nesta pasta.

## Modelo adotado

| Arquivo | Modelo | Classes | Download |
| :--- | :--- | :--- | :--- |
| `yolo11ft.pt` | YOLOv11 fine-tuned (gado aéreo) | `0 = cow` | <https://github.com/xxilytoo/cattle_graze_model> |

```powershell
# Windows / PowerShell (a partir da raiz do projeto)
New-Item -ItemType Directory -Force -Path models | Out-Null
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/xxilytoo/cattle_graze_model/main/yolo11ft.pt" -OutFile "models/yolo11ft.pt"
```

O caminho já é o padrão em `config.MODEL_WEIGHTS`. Se o arquivo não existir, o
`main.py` avisa com instruções em vez de estourar um erro genérico do Ultralytics.

## Alternativas testadas

| Arquivo | Modelo | Observação |
| :--- | :--- | :--- |
| `cow_weight_v1.pt` | YOLOv8 (ICAERUS, nadir 30–100 m) | ~13 detecções/frame no vídeo denso; fraco no oblíquo |
| `cow_weight_v2_12.pt` | YOLOv8m (ICAERUS v2, recortes 640×640) | 0 detecções no vídeo oblíquo; só funciona bem em nadir |

```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ICAERUS-EU/UC3_Livestock_Monitoring/main/models/cow_detection/cow_detection_v1/cow_weight_v1.pt" -OutFile "models/cow_weight_v1.pt"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ICAERUS-EU/UC3_Livestock_Monitoring/main/models/cow_detection/cow_detection_v2/cow_weight_v2_12.pt" -OutFile "models/cow_weight_v2_12.pt"
```

## Fallback genérico (COCO)

`yolov8n.pt` é baixado automaticamente pelo Ultralytics na primeira execução:

```bash
python main.py --input video.mp4 --output saida.mp4 --model yolov8n.pt
```

Nesse caso, ajuste `CLASSES_OF_INTEREST = [18, 19]` (sheep + cow) em `config.py`.
