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

Pipeline completo: `gado.webm` **23 → 128** confirmados; `video-gado.mp4` 72 → 59 (calibração do tracker de 04/11/2026 — ver seção abaixo). A fragmentação de IDs ainda é o gargalo.

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

## 🔬 Validação e calibração do rastreador (04/11/2026)

O gargalo da contagem não era o detector: era o **limiar de associação do ByteTrack**. Com `MINIMUM_MATCHING_THRESHOLD=0.70` (exige IoU > 0,30 entre a caixa nova e a prevista), um bovino pequeno e parecido com o vizinho perdia a associação por 1 frame e ganhava um ID novo. O `supervision` usa `1 - IoU` como custo, então **aumentar** o limiar **afrouxa** a associação.

Varredura em duas janelas de 150 frames (`python tune.py`), com as detecções cacheadas:

| `matching` | confirmados | IDs | fragmentação | IDs-lixo (≤3 frames) | ativos/frame |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0,70 (antigo) | 29 | 67 | **2,31** | 10 | 13,1 |
| 0,80 | 31 | 44 | 1,45 | 4 | 14,5 |
| **0,90 (adotado)** | 27 | 36 | **1,33** | 1 | 14,9 |
| 0,95 | 25 | 36 | 1,44 | 1 | 14,9 |

Resultado nos vídeos completos (mesmo modelo, mesmos limiares de detecção, mesma velocidade):

| Vídeo | Antes (0,70) | Depois (0,90) |
| :--- | :--- | :--- |
| `video-gado.mp4` | 69 confirmados / 180 IDs / frag 2,61 | **59 / 103 / 1,75** |
| `gado.webm` | 133 confirmados / 439 IDs / frag 3,30 | **128 / 182 / 1,42** |

Os IDs caíram 43% e 59%: o mesmo animal deixou de ser contado 3 vezes. A contagem final mudou pouco (‑4% no denso, ‑14% no oblíquo) porque os IDs duplicados antes **também** confirmavam.

### Ferramentas de medição

```bash
# 1. Relatório de fragmentação por ID (CSV em validation/)
python validate.py report --input video-gado.mp4 --limit 150

# 2. Frames-chave anotados para contagem manual (referência)
python validate.py frames --input video-gado.mp4 --limit 150 --pick-best 3

# 3. Registrar a contagem manual feita na imagem
python validate.py gt --video video-gado.mp4 --frame 15 --count 24

# 4. Grid de parâmetros usando só o cache de detecções (sem rodar o YOLO de novo)
python tune.py --input video-gado.mp4 --limit 150 --matchings 0.70,0.80,0.90,0.95
```

> **Atenção metodológica:** `confirmados` não é monotônico em relação à fragmentação. Numa janela em que um animal fica no quadro mas o ID quebra em pedaços menores que 15 frames, ele **não** confirma (subcontagem); quando os pedaços são maiores, ele confirma duas vezes (supercontagem). Medimos os dois efeitos: na janela `[450:600]` do `video-gado.mp4`, o pico simultâneo é 23 e o total confirmado era 16 com `matching=0.70` e 23 com `0.90`. Por isso a **contagem manual de referência** (passo 2–3) é obrigatória antes de afirmar precisão.

**Ajuste fino sem editar código** (novas flags do `main.py`):

```bash
python main.py --input video-gado.mp4 --output saida.mp4 \
  --track-act 0.30 --min-match 0.90 --streak-tol 0
# --min-match maior  -> aceita IoU menor na associação (menos fragmentação)
# --streak-tol N     -> tolera N frames ausentes antes de zerar a persistência
```

**Parâmetro `STREAK_GAP_TOLERANCE`:** a persistência ainda é consecutiva (`0`). Tolerar 2 frames ausentes recupera animais que "piscam" (na janela `[450:600]`, 16 → 23 confirmados, igual ao pico), mas aumenta ~7% o total na janela `[0:150]` — sem referência manual não dá para dizer qual dos dois está certo. Ficou como experimento (`--streak-tol`), não como padrão.

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
