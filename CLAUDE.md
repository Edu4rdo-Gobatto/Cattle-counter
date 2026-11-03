# CLAUDE.md — Contador Aéreo de Bovinos (Drone/UAV)

Contexto e regras para qualquer agente de IA que trabalhe neste repositório. Siga estas diretrizes em toda geração de código, refatoração ou explicação.

---

## 1. Projeto

* **Objetivo:** detectar, rastrear e contar bovinos em vídeos de drone (visão nadir 90° ou oblíqua 45°), em Python.
* **Contexto:** projeto acadêmico de graduação em Sistemas de Informação. A avaliação valoriza código limpo, modularidade e aplicação prática de algoritmos consolidados de tracking.
* **Repositório:** https://github.com/Edu4rdo-Gobatto/Cattle-counter (branch `main`).

## 2. Regras inegociáveis

1. **Zero rotulagem manual.** É PROIBIDO instruir o usuário a desenhar bounding boxes, anotar ou treinar modelos do zero. Use pesos pré-treinados (`.pt`) ou datasets que já venham 100% anotados no padrão YOLO.
2. **Entrega pragmática.** O pipeline deve rodar assim que houver um `.pt` e um `.mp4`.
3. **Sem código monolítico.** Responsabilidades separadas em `config`, `detector`, `tracker` e `counter`. Nada de scripts únicos de 500 linhas.
4. **Eficiência.** Ler vídeo frame a frame (gerador ou `cv2.VideoCapture`), nunca carregar o vídeo inteiro na RAM. Evitar cópias desnecessárias de arrays NumPy no loop.
5. **Debate antes de grandes mudanças.** Antes de refatorar ou mudar a lógica de contagem, exponha os trade-offs e confirme a dinâmica de voo com o desenvolvedor.
6. **Caminhos e parâmetros só no `config.py`.** Nunca espalhar caminhos absolutos ou thresholds pelo código.
7. **Estilo:** tipagem com `typing`, sintaxe clara e comentários objetivos em português.
8. **Commits:** mensagens em português, **sem** linhas `Co-Authored-By` ou qualquer atribuição a IA.

## 3. Stack

| Componente | Ferramenta | Finalidade |
| :--- | :--- | :--- |
| Linguagem | Python 3.10+ | Base |
| Detecção | `ultralytics` (YOLOv8/v11) | Inferência por frame |
| Rastreamento | `supervision.ByteTrack` | IDs persistentes (filtro de Kalman) |
| Lógica espacial | `supervision.LineZone` / `PolygonZone` | Contagem por cruzamento/presença |
| Vídeo | `opencv-python` | Leitura, escrita e desenho |

Ambiente validado (`.venv`, Windows, CPU AMD Ryzen 7 5700U): ultralytics 8.4.170, supervision 0.30.6, opencv 5.0.0, numpy 2.4.6, torch 2.14.1+cpu.

---

## 4. Estado atual do código (fonte da verdade)

Os módulos estão **na raiz** do projeto:

| Arquivo | Papel |
| :--- | :--- |
| `config.py` | Caminhos, device, thresholds, parâmetros do ByteTrack, cores do HUD |
| `detector.py` | `CattleDetector`: YOLO → `sv.Detections`, filtro por classe, threads da CPU |
| `tracker.py` | `CattleTracker`: wrapper do `sv.ByteTrack` |
| `counter.py` | `CattleCounter`: contagem ativa + IDs únicos com filtro de persistência + LineZone opcional |
| `main.py` | Orquestra o pipeline, desenha o HUD e grava o vídeo anotado (`argparse`: `--input --output --model --enable-linezone`) |
| `test_pipeline.py` | 5 testes `unittest` (contador, persistência, ruído, giro 360°, HUD) |
| `generate_test_video.py` | Gera um vídeo sintético (elipses sobre fundo verde) |
| `run.bat` | Roda `main.py` com `input_video.mp4` ou, na falta dele, `test_drone_video.mp4` |

**Lógica de contagem (diferencial do projeto):** um `tracker_id` só entra no "TOTAL CONFIRMADO" depois de aparecer em `MIN_FRAMES_PERSISTENCE` (15) frames. Isso filtra falsos positivos (pedras, sombras) e torna a contagem imune a giros de 360° do drone, porque os mesmos IDs não são somados de novo. A LineZone (in/out) é opcional (`ENABLE_LINEZONE`) e serve para voos retilíneos em corredor.

**Parâmetros principais (`config.py`):** `MODEL_WEIGHTS=models/yolo11ft.pt` (YOLOv11 especializado, 1 classe: `cow`), `CLASSES_OF_INTEREST=[0]`, `USE_SLICING=False` (`SLICE_SIZE=640`, `SLICE_OVERLAP=128`), `CONFIDENCE_THRESHOLD=0.25`, `IOU_THRESHOLD=0.60` (calibrados para animais distantes/agrupados), `TRACK_ACTIVATION_THRESHOLD=0.30`, `LOST_TRACK_BUFFER=45`, `MINIMUM_MATCHING_THRESHOLD=0.70`, `INFERENCE_IMAGE_SIZE=640`, `DEVICE="cpu"`, `CPU_NUM_THREADS` = metade de `os.cpu_count()`, `SHOW_WINDOW=False`. Para voltar ao COCO: `MODEL_WEIGHTS="yolov8n.pt"` e `CLASSES_OF_INTEREST=[18, 19]`.

### Como executar
```bash
.\run.bat                      # Windows: demo com janela ao vivo (./run.sh no Linux)
.venv\Scripts\activate
python -m unittest test_pipeline -v
python main.py --input video-gado.mp4 --output saida.mp4 [--show] [--slicing] [--conf 0.25] [--iou 0.60] [--imgsz 640]
```
O roteiro da apresentação em sala está em `APRESENTACAO.md`.

### Resultados dos testes (02/11/2026)
* `unittest`: **7/7 OK** (contador com persistência consecutiva, SAHI e HUD).
* **Vídeo sintético:** roda, mas conta 0. Modelos de gado não detectam as elipses. Esse vídeo não serve para validar a detecção.
* **`video-gado.mp4`** (real, TikTok @dezodrones, 576x1024 vertical, 20s, drone baixo e oblíquo, gado Nelore branco):
  * **Linha de base COCO (`yolov8n`, `[18, 19]`):** **72 confirmados**, ~11 FPS. Os animais próximos são detectados (conf 0,5–0,9), os distantes no fundo não. Valor provavelmente **acima do real**: 121 IDs criados. (~20 frames finais são a vinheta do TikTok, sem bois.)
  * **Modelo especializado:** **64 confirmados**, 175 IDs, ~6 FPS.

### Modelo especializado integrado (02/11/2026)
* **Modelo adotado:** `models/yolo11ft.pt` — YOLOv11 do repo `xxilytoo/cattle_graze_model` (1 classe: `cow`), baixado pronto da internet (transfer learning, sem rotulagem manual).
* **Benchmark (40 frames amostrados, conf 0.35, CPU):** `video-gado.mp4` 12,5 → **13,3 detecções/frame** (0,37 → 0,19 s/frame); `gado.webm` 2,3 → **18,9 detecções/frame**.
* **Pipeline completo:** `gado.webm` **23 → 133 confirmados** (104 → 439 IDs); `video-gado.mp4` **72 → 69 confirmados** (146 → 180 IDs).
* **Calibração distantes/agrupados (02/11/2026):** `conf 0.35→0.25` e `iou 0.45→0.60` (mesma velocidade). Benchmark: +16% detecções/frame no `gado.webm` (18,2 → 21,2) e +14% no `video-gado.mp4` (12,5 → 14,2); contagem final +11% e +8%. `imgsz=960` rende mais (+34% no denso), mas dobra o tempo e ficou opcional via `--imgsz`.
* **SAHI (`--slicing`) no `gado.webm`:** 18,9 → 20,9 detecções/frame, mas ~1,9 s/frame (≈10x mais lento); desligado por padrão.
* **SAHI (`--slicing`) no `video-gado.mp4` (pipeline completo, limiares antigos):** 66 confirmados (vs 64 sem SAHI), 227 IDs, fragmentação 3,44 e 2,58 FPS. Ganho pequeno para ~4x mais tempo — por isso fica opcional.
* **Contador:** persistência agora é **consecutiva** (o streak zera quando o ID some do frame); total de frames vistos continua como diagnóstico.
* **Conclusão:** o modelo especializado resolveu a subcontagem do rebanho denso (5x). A fragmentação de IDs (2,7–3,3) continua sendo o gargalo do `video-gado.mp4`.

### Já resolvido (commit `ffd5699` e ajustes seguintes)
* ~~`CLASSES_OF_INTEREST = [18, 19]`~~ (histórico COCO): agora `[0]` com o modelo especializado.
* `supervision` travada em `<0.31.0` (o `sv.ByteTrack` será removido na 0.31) e `numpy` sem teto.
* `counts` inicializado antes do loop (sem `NameError` com vídeo vazio).
* HUD responsivo: quebra linha em vídeo vertical.
* `CPU_NUM_THREADS` automático e sem menção fixa à CPU.
* Janela ao vivo: `--show` / `SHOW_WINDOW` (espaço pausa, F tela cheia, Q/ESC sai).
* `run.bat` (Windows) e `run.sh` (Linux): aceitam um vídeo como argumento ou usam o primeiro que existir entre `input_video.mp4`, `video-gado.mp4` e `gado-teste.mp4`, sempre com `--show`.

### Pendências conhecidas
1. **Troca de IDs com a câmera em movimento** infla o total (121 IDs criados para 72 confirmados no `video-gado.mp4`). O resumo final do `main.py` agora imprime "IDs Criados pelo Tracker" e a razão de fragmentação para medir isso.
2. **Rebanho denso + drone alto + motion blur** (`gado.webm`, 1080x1920): o `yolov8n` subcontava fortemente (23 confirmados para 60+ animais visíveis num quadro). Com o modelo especializado `yolo11ft.pt` a contagem subiu para **120 confirmados** (~8x mais detecções/frame), mas a fragmentação ainda infla o total (398 IDs). Ajustes de limiar no COCO já testados (imgsz 1280, conf 0.25 etc.) não foram adotados: o limite era o modelo.
3. **Modelo genérico:** ~~trocar por um `best.pt` do Roboflow~~ **Resolvido:** adotado `models/yolo11ft.pt` (YOLOv11, `xxilytoo/cattle_graze_model`, classe 0 = `cow`), baixado pronto (transfer learning, sem rotulagem).
4. **Migrar o tracker** antes de atualizar a `supervision` para a 0.31+.
5. **Linha solta `aa`** no `.gitignore` (inofensiva).

### Próximos passos estruturais
* **SAHI (implementado, opcional):** `--slicing` fatia o frame em janelas 640x640 com overlap e funde as caixas com NMS. No `gado.webm` subiu de 18,9 para 20,9 detecções/frame, mas custa ~1,9 s/frame (≈10x mais lento); por isso `USE_SLICING=False` por padrão.
* **Modelo especializado (implementado):** adotado `models/yolo11ft.pt` (YOLOv11, repo `xxilytoo/cattle_graze_model`), baixado pronto — sem rotulagem manual. Benchmark: `gado.webm` 2,3 → 18,9 detecções/frame; `video-gado.mp4` 12,5 → 13,3 (e 2x mais rápido). Fontes alternativas testadas: Roboflow "aerial cows" (mAP@50 86.2%), ICAERUS cow YOLOv8 nadir (P 0.93/R 0.85 no v2, mas falhou no vídeo oblíquo).
* **Compensação de movimento da câmera (CMC):** estimar a homografia entre frames consecutivos e corrigir a predição do Kalman antes do matching. Ataca a causa raiz da troca de IDs (fragmentação 2,7–3,3 ainda observada).

---

## 5. Arquitetura alvo (migração futura)

A estrutura oficial desejada move os módulos para `src/`. **Não migre sem combinar com o desenvolvedor** (regra 5), e preserve a lógica de persistência do `counter.py` atual.

```plaintext
contador-bovinos/
├── .gitignore
├── requirements.txt         # Dependências pinadas
├── main.py                  # Maestro do pipeline
├── models/best.pt           # Pesos pré-treinados (YOLO)
├── data/
│   ├── input/               # Vídeos brutos (.mp4)
│   ├── output/              # Vídeos anotados
│   └── datasets/            # Datasets anotados (opcional)
└── src/
    ├── __init__.py
    ├── config.py
    ├── detector.py
    ├── tracker.py
    └── counter.py
```

`.gitignore` alvo: `.venv/`, `venv/`, `ENV/`, `__pycache__/`, `*.py[cod]`, `models/*.pt`, `models/*.onnx`, `data/datasets/`, `data/input/`, `data/output/`, `.vscode/`, `.idea/`. O atual já ignora `*.pt`, `*.mp4`, `*.avi`, `*.mov`, `*.mkv` e `.env`.

## 6. Pipeline

```plaintext
Vídeo (.mp4) ─► leitura frame a frame (OpenCV)
  ─► CattleDetector (YOLO)      : boxes + confiança, filtradas por classe
  ─► CattleTracker (ByteTrack)  : tracker_id via filtro de Kalman
  ─► CattleCounter              : ativos no quadro, IDs confirmados, LineZone opcional
  ─► Anotadores                 : Trace + Box + Label + LineZone + HUD
  ─► cv2.VideoWriter (vídeo anotado)
```

## 7. Perspectivas a equilibrar ao propor mudanças

* **Arquiteto de Visão Computacional:** compatibilidade entre a saída da YOLO e o `supervision`. Manter `conf` em 0,3–0,4 e `iou` em 0,45, evitando caixas fantasmas ou duplicadas.
* **Performance:** baixa latência, memória previsível, sem cópias desnecessárias, suporte a CUDA/MPS com fallback eficiente para CPU.
* **QA de domínio (agro/aéreo):**
  * **Movimento do drone (vento/translação):** linhas virtuais fixas na tela falham sem compensação de movimento da câmera (CMC). A movimentação também causa troca de IDs, o que infla o total.
  * **Oclusão** (árvores, sombras): `lost_track_buffer` de 30 a 60 frames.
  * **Aglomeração:** animais muito próximos podem ter as caixas fundidas.
  * **Raças:** o Nelore branco é classificado como `sheep` pelo COCO.

## 8. Obtenção de dados (sem rotulagem manual)

**Pesos (`best.pt`), caminho padrão.** Baixar um modelo pronto em vez de treinar.
1. Acessar universe.roboflow.com e buscar `aerial cattle detection`, `drone livestock` ou `uav grazing cows`.
2. Escolher um projeto com o selo "Model Available" (ex.: Aerial Cattle Detection, Cattle-Monitoring-UAV, Livestock-Detection).
3. Na aba Model, usar Download Weights no formato YOLOv8 PyTorch.
4. Salvar o arquivo e rodar com `--model caminho/best.pt`.

**Dataset anotado.** Só se a faculdade exigir que ele seja apresentado.
```python
import os
from roboflow import Roboflow  # pip install roboflow

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])  # nunca commitar a chave
project = rf.workspace("<workspace>").project("<projeto>")  # IDs reais do snippet do Roboflow
dataset = project.version(1).download("yolov8", location="data/datasets/cattle_dataset")
```

**Vídeos de teste.**
```bash
pip install yt-dlp
yt-dlp -f "bestvideo[ext=mp4]" "<URL>" -o "input_video.mp4"
```
Termos de busca: `drone cattle herd 4k`, `aerial view cows grazing`, `livestock drone footage`. Prefira drone baixo e câmera oblíqua quando usar o modelo COCO genérico.

## 9. Argumentos para a apresentação acadêmica

1. **Modelos pré-treinados (transfer learning):** backbones treinados em milhares de imagens generalizam melhor que um modelo treinado do zero com poucas imagens, que sofreria overfitting. O esforço do projeto vai para a camada de rastreamento e contagem.
2. **ByteTrack em vez de contar caixas por frame:** contar por frame repete o mesmo animal várias vezes. O ByteTrack mantém um ID por animal (Kalman, aproveitando inclusive detecções de baixa confiança) e o filtro de persistência descarta ruídos.
3. **Camadas separadas:** a interface (OpenCV/HUD), a inferência (`detector`) e as regras de negócio (`counter`) ficam isoladas. Trocar o modelo ou trocar a linha por um polígono afeta um único arquivo.
4. **Achado prático:** o modelo genérico confunde Nelore com ovelha. Isso justifica usar um modelo especializado em gado aéreo.
