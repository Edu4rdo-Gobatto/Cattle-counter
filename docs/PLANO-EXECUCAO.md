# Plano de execução — sem vídeo oficial + dataset (dentro do repo)

Contexto: `video-gado.mp4` sumiu. Sem vídeo oficial, na apresentação serão testados vários vídeos ao vivo. `gado.webm` (1080x1920, 644 frames decodificados) é só vídeo de dev. Modelo atual: `models/yolo11ft.pt`. Testes: 10/10 OK. `train.py`, `pseudo_label.py`, `download_dataset.py`, `notebooks/treino_colab.ipynb`, `config.py §7 (TRAIN_*/PSEUDO_*)` já existem (Etapa 0 do T2 feita em 27/11).

## 1. Sem vídeo fixo

1. Não cravar número único no slide. Resultado vira faixa + protocolo, não "66".
2. Manter `gado.webm` como referência de regressão, mas apresentação leva 2-3 vídeos anotados como plano B.
3. `run.bat`/`run.sh` já aceitam argumento e fallback para `gado.webm`. Na hora: `run.bat caminho\do\video.mp4 --show`.
4. `--end-frame` vira argumento de hora para cortar vinhetas, nunca default fixo.
5. Thresholds atuais (`conf 0.15`, `iou 0.60`, `matching 0.90`, `streak-tol 0`, filtro objeto fixo 10px/100f) foram calibrados no oblíquo perdido — tratar como ponto de partida geral, validar por erro quando der.

## 2. Dataset (sem chave — via HuggingFace)

Escolha: `Francesco/aerial-cows` (espelho RF100 `aerial-cows`, CC BY 4.0, 1084 train / 299 valid / 340 test, 640px).
Motivo: mesma base do "aerial cows" da Roboflow (mAP@50 86,2% de referência), 1 classe `cow`, sem precisar de `ROBOFLOW_API_KEY`.
Reserva: ICAERUS-EU/UC3_Livestock (nadir puro) e ZIP original da Roboflow (esse sim exige chave).

Sobre a API-KEY: não dá para buscar com e-mail. A Roboflow só gera a chave dentro da conta logada. Passo manual de 2 min, só se quiser a versão ZIP original:
1. Conta em universe.roboflow.com.
2. Projeto aerial cows → Download Dataset → formato YOLOv8 → copiar snippet com `ROBOFLOW_API_KEY`.
3. `export ROBOFLOW_API_KEY="..."` sem commitar. Nunca no código.

Execução sem chave:
- `uv pip install --python .venv/bin/python datasets pillow` (venv é uv, sem pip interno).
- `.venv/bin/python download_dataset.py` → `data/datasets/aerial-cows/` + `data.yaml` (conversão COCO→YOLO, classe única `cow`).
- Crédito obrigatório: "aerial-cows from Roboflow 100 (Roboflow Universe), CC BY 4.0."
- `yolo val model=models/yolo11ft.pt data=data/datasets/aerial-cows/data.yaml split=test imgsz=640` → linha "antes".

## 3. Feito em 27/11

- `gado.webm [0:150] conf 0.15`: 12,9/frame, pico 31, 41 confirmados / 66 IDs / frag 1,61 (report).
- Grid tracker: 0.70 → 31/122/3,94; 0.80 → 41/87/2,12; **0.90 → 41/66/1,61**. Mesmo padrão do vídeo perdido: 0.90 vence.
- Frames p/ contagem humana: `validation/gado_f00018.png` (29 det), `f00067.png` (13), `f00147.png` (42).
  Contagens registradas: 18→38, 67→12, 147→47 (pico humano 47) e total distinto 100 em [0:644]. Report [0:150]: pico detector 31 (66% do visível), 41 confirmados / 66 IDs / frag 1,61, erro -13% vs pico. Grid: 0.90 vence (41, -13%, 1,61).
  Full [0:644] conf 0.15: 19,7/frame, pico 39, **129 confirmados / 184 IDs / frag 1,43, erro +29% vs 100**. Grid full: 0.70 → +38%/3,27; 0.80 → +45%/1,87; **0.90 → +29%/1,43** (melhor nos dois critérios).
- Dataset: `data/datasets/aerial-cows/` completo via HF (1084 train / 299 valid / 340 test + `data.yaml`, CC BY 4.0).
- Linha base `yolo11ft.pt` no test (340 imgs, 2823 inst, imgsz 640, CPU): **P 0,216 / R 0,143 / mAP50 0,061 / mAP50-95 0,019**. O resultado justificou testar o fine-tune A; a medição no vídeo mostrou que o ganho no split público não se transferiu para `gado.webm`.
- Fine-tune A do Colab confirmado localmente em `models/best_a.pt`: no mesmo split test, **P 0,760 / R 0,607 / mAP50 0,652 / mAP50-95 0,282**.
- `tune.py --model models/best_a.pt` mediu `gado.webm [0:644]`: **3,0 detecções/frame** (pico bruto 31), **0 confirmados / 12 IDs** em matching 0,90 e erro **−100%** contra o total humano 100. No grid: matching 0,70 → 0 / 16; 0,80 → 1 / 15; 0,90 → 0 / 12. CSV e cache têm sufixo `_mbest_a`.
- A inspeção dos frames de referência mostra a queda de confiança fora do domínio: `yolo11ft` detectou 29/13/42 caixas nos frames 18/67/147; A detectou 0/22/5 com conf 0,15. O pico de caixas não é a mesma métrica que os animais simultâneos aceitos pelo tracker.
- Decisão da Etapa 5 para A: manter `yolo11ft.pt` como modelo padrão. A vence no teste público, mas perde no erro da contagem do vídeo.
- A+B: gerados 22 pseudo-rótulos a cada 30 frames de `gado.webm`, total de 362 caixas em 20 frames não vazios. A tentativa de treino local foi encerrada pelo limite de 120 s após 15/70 batches da primeira época. Não há pesos finais nem métricas de A+B; retomar no Colab ou em uma execução com várias horas disponíveis.
- Curvas PR e matriz de confusão de A foram salvas por `yolo val` em `runs/detect/val-3/`. O `results.png` do treino A não veio do Colab.

## 4. Próximos passos

1. Retomar o treino A+B no Colab ou em uma execução local com tempo suficiente. O caminho de saída foi ajustado para `runs/train/ab/`.
2. Avaliar `best.pt` de A+B no split test e em `gado.webm [0:644]`, mantendo conf 0,15, IoU 0,60, matching 0,90 e tol 0. Não reutilizar o cache do `yolo11ft`.
3. Preencher a linha A+B em `docs/PLANO-TREINAMENTO.md`. A decisão do modelo padrão só muda se mAP e erro de contagem melhorarem.
4. Obter do Colab os gráficos do treino A, se ainda estiverem disponíveis. Os gráficos de validação local estão em `runs/detect/val-3/`.
5. Fazer uma avaliação de referência em outro vídeo antes de apresentar uma faixa geral. `gado.webm` é apenas vídeo de desenvolvimento.

## Fora de escopo (regras do repo)

- Zero rotulagem manual: sem desenhar caixas.
- Sem commitar `*.pt`, vídeos, `runs/`, `cache/`, `data/pseudo/` ou chave de API.
- Sem refatorar contagem/tracker sem debate (regra 5).
- Parâmetros novos só em `config.py`.
