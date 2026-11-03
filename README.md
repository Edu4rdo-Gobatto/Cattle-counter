# 🛰️ Sistema de Detecção, Rastreamento e Contagem Aérea de Bovinos

Aplicação em Python de alta performance para monitoramento e contagem automatizada de rebanho bovino através de filmagens aéreas de drones.

---

## 👥 Decisões da Squad de Engenharia

O comitê técnico multidisciplinar estruturou a aplicação para o hardware **AMD Ryzen 7 5700U** em modo **offline**, considerando que o drone realiza curvas e giros de 360°:

1. **Imunidade a Giros 360° (Agente 3 - QA):**
   * Em vez de depender exclusivamente de uma linha estática na tela (que causaria contagens repetidas a cada volta do drone sobre o pasto), o sistema conta com um **Totalizador de IDs Únicos com Filtro de Persistência** (`MIN_FRAMES_PERSISTENCE`).
   * Um animal só é registrado no totalizador se for rastreado continuamente por pelo menos 15 frames, filtrando falsos positivos efêmeros (copas de árvores, pedras, sombras).
   * O HUD também exibe a métrica **NO QUADRO**, indicando quantos bovinos estão visíveis naquele exato segundo de filmagem.

2. **Otimização Ryzen 7 5700U (Agente 2 - Performance):**
   * Configuração de multithreading do PyTorch (`torch.set_num_threads(8)`) para paralelização nos 8 núcleos da CPU Zen 2.
   * Utilização de um modelo especializado em gado aéreo (`yolo11ft.pt`, YOLOv11) para inferência rápida sem GPU dedicada.
   * Uso de estruturas vetorizadas do `supervision` para cálculo de IoU e associação do ByteTrack.

3. **Arquitetura Modular Desacoplada (Agente 1 - Lead Architect):**
   * `config.py`: Centraliza todos os hiperparâmetros, diretórios e limiares de confiança.
   * `detector.py`: Encapsula a inferência do YOLO e converte predições em objetos `sv.Detections`.
   * `tracker.py`: Gerencia o algoritmo ByteTrack com predição de Filtro de Kalman e buffer de oclusão.
   * `counter.py`: Calcula a contagem ativa, o acúmulo de IDs únicos e suporte opcional a `LineZone`.
   * `main.py`: Executa o pipeline de vídeo, renderiza o HUD superior profissional e salva o vídeo anotado.

---

## 🧠 Modelos Prontos para Download (Treinamento Já Feito)

> **Contexto:** o `yolov8n.pt` genérico (COCO) já foi avaliado e não distingue bem o gado voador. Pesquisas na web trouxeram modelos **prontos** (transfer learning) que podem substituir o backbone sem nenhum treinamento manual.

| Fonte | Modelo | Métricas / Observação | Relevância |
| :--- | :--- | :--- | :--- |
| Roboflow Universe | "aerial cows" (Roboflow 100 / Livestalk) | mAP@50 **86.2%**, Precisão **88.8%**, Recall **81.3%** (1 classe: `cow`) | Alta |
| ICAERUS-EU/UC3_Livestock_Monitoring | Cow detection YOLOv8 (vista nadir, ~100 m) | Modelos `cow_detection_v1` / `cow_detection_v2` | Alta |
| Hugging Face – `rmayormartins/yolov5-farm-cattle` | YOLOv5 (`bestyolo5.pt`) | Foco em *cattle counter* | Média |
| Hugging Face – `xxilytoo/cattle_graze_model` | YOLOv11 (`yolo11ft.pt`) | Detector de gado em voo | Média |
| Hugging Face – `cpraschl/bambi-models` | YOLOv26 (drone RGB/thermal) | mAP 0.946 (wildlife) | Baixa-média |
| Hugging Face – `dronefreak/visdrone-yolo11s` | YOLO11s (VisDrone) | Foco em pedestres/animais | Baixa |

### ✅ Modelo adotado: `yolo11ft.pt` (YOLOv11 — cattle_graze_model)

Benchmark local (40 frames amostrados, `conf=0.35`, `iou=0.45`, CPU):

| Vídeo | yolov8n (COCO) | **yolo11ft** | yolo11ft + SAHI |
| :--- | :--- | :--- | :--- |
| `video-gado.mp4` (oblíquo) | 12,5 detecções/frame | **13,3** (0,19 s/frame) | 14,1 (0,49 s/frame) |
| `gado.webm` (denso) | 2,3 detecções/frame | **18,9** | 20,9 (1,89 s/frame) |

Pipeline completo: `gado.webm` **23 → 133** confirmados; `video-gado.mp4` 72 → 69 (com `--slicing`: 66, nos limiares antigos). A fragmentação de IDs ainda é o gargalo.

**Onde fica:** `models/yolo11ft.pt` — já configurado em `config.MODEL_WEIGHTS`. Instruções de download em [`models/README.md`](models/README.md).

**Calibração para distantes/agrupados:** `CONFIDENCE_THRESHOLD=0.25` e `IOU_THRESHOLD=0.60` (mesma velocidade, +8–11% na contagem). Ajuste fino sem editar código:

```bash
python main.py --input gado.webm --output saida.mp4 --conf 0.25 --iou 0.60 --imgsz 640
# --conf menor  -> pega animais distantes (e mais ruído)
# --iou maior   -> preserva caixas de animais agrupados
# --imgsz maior -> detecta melhor animais pequenos (mais lento)
```

**Como testar:**

```bash
python main.py --input video-gado.mp4 --output saida.mp4                 # modelo especializado (padrão)
python main.py --input gado.webm --output saida_sahi.mp4 --slicing       # SAHI (mais lento, mais detecções)
```

> Para voltar ao COCO: `--model yolov8n.pt` e ajustar `CLASSES_OF_INTEREST=[18, 19]` no `config.py`.

**Linha de base antiga:** `video-gado.mp4` → 72 confirmados / 146 IDs; `gado.webm` → 23 confirmados / 104 IDs.

---

## 🚀 Instalação das Dependências

Abra o terminal (Prompt de Comando ou PowerShell) na pasta do projeto e instale as dependências:

```bash
pip install -r requirements.txt
```

> **Dica para CPU:** Para garantir a versão do PyTorch mais leve e otimizada para CPU:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install ultralytics supervision opencv-python tqdm
> ```

### Baixar o modelo (uma vez)

Os pesos `*.pt` não são versionados. Baixe o modelo adotado para `models/`:

```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/xxilytoo/cattle_graze_model/main/yolo11ft.pt" -OutFile "models/yolo11ft.pt"
```

Mais opções (alternativas e fallback COCO): [`models/README.md`](models/README.md).

---

## 🎬 Como Executar

### 1. Execução Básica (Padrão)
Coloque o vídeo gravado do drone com o nome `input_video.mp4` dentro da pasta do projeto e execute:

```bash
python main.py
```

O vídeo final anotado será salvo como `output_annotated_video.mp4`.

---

### 2. Execução com Argumentos Customizados
Você pode especificar o caminho de entrada, saída e modelo através da linha de comando:

```bash
python main.py --input "C:/caminho/do/meu_video.mp4" --output "C:/caminho/resultado.mp4"
```

---

### 3. Ativar LineZone (Para voos retilíneos em corredores/porteiras)
Se você tiver um trecho de vídeo onde o voo é puramente retilíneo ou o gado passa por uma porteira/corredor, adicione a flag:

```bash
python main.py --enable-linezone
```

---

## 📊 Elementos Gráficos no Vídeo Anotado

* **Bounding Box Azul Celeste:** Delimitação nítida do bovino detectado.
* **Rastro Laranja (Trace):** Histórico visual do deslocamento do animal nos últimos frames.
* **ID & Confiança:** Identificador único do animal persistido pelo ByteTrack (`#ID Confiança`).
* **Painel Superior (HUD):**
  * **NO QUADRO:** Animais ativos no frame atual.
  * **TOTAL CONFIRMADO:** Contagem cumulativa validada pelo filtro de persistência.
  * **Frame / FPS:** Telemetria de desempenho em tempo real do processador Ryzen.
