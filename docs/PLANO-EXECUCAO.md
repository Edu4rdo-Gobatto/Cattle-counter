# Plano de execução — sem vídeo oficial + dataset (dentro do repo)

Contexto: `video-gado.mp4` sumiu. Sem vídeo oficial — na apresentação serão testados vários vídeos ao vivo. `gado.webm` (1080x1920, 644 frames decodificados) é só vídeo de dev. Modelo atual: `models/yolo11ft.pt`. Testes: 9/9 OK. `train.py`, `pseudo_label.py`, `download_dataset.py`, `notebooks/treino_colab.ipynb`, `config.py §7 (TRAIN_*/PSEUDO_*)` já existem (Etapa 0 do T2 feita em 27/11).

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
- Falta humana: `validate.py gt --video gado.webm --frame <N> --count <C>` (3 frames) + `--total X` p/ erro %.
  Feito 27/11: 18→38, 67→12, 147→47 (pico humano 47) + total distinto 100 em [0:644]. Report [0:150]: pico detector 31 (66% do visível), 41 confirmados / 66 IDs / frag 1,61, erro -13% vs pico. Grid: 0.90 vence (41, -13%, 1,61).
  Full [0:644] conf 0.15: 19,7/frame, pico 39, **129 confirmados / 184 IDs / frag 1,43, erro +29% vs 100**. Grid full: 0.70 → +38%/3,27; 0.80 → +45%/1,87; **0.90 → +29%/1,43** (melhor nos dois critérios).
- Dataset: `data/datasets/aerial-cows/` completo via HF (1084 train / 299 valid / 340 test + `data.yaml`, CC BY 4.0). `yolo val` base rodando em background.
- Linha base `yolo11ft.pt` no test (340 imgs, 2823 inst, imgsz 640, CPU): **P 0,216 / R 0,143 / mAP50 0,061 / mAP50-95 0,019**. Baseline honesto e baixo: domínio RF100 ≠ domínio do fine-tune — é exatamente o argumento para treinar A/A+B no Colab.

## 4. Próximos passos (nesta ordem)

1. Confirmar `gado.webm` abre e `models/yolo11ft.pt` existe.
2. Protocolo genérico multi-vídeo:
   - `validate.py frames --input <video> --limit N --pick-best 3`
   - contagem humana 3 frames + total distinto → `validate.py gt --video <nome> --frame N --count C` e `--total X --end Y`
   - `validate.py report --input <video>` + `tune.py --input <video> --matchings 0.70,0.80,0.90 --tolerances 0`
   - ranking por erro vs humano, empate → menor frag.
3. Dataset: terminar `download_dataset.py`, rodar `yolo val` base.
4. Treino (Colab T4) — feito 27/11, modelo A:
   - Treino: 30 épocas, valid mAP50 0,495; test (340) P 0,760 / R 0,607 / mAP50 0,652 (base era 0,061).
   - Trazer `runs/detect/runs/train/a/weights/best.pt` do Colab para `models/best_a.pt` e avaliar no `gado.webm`.
   - Pseudo A+B (opcional): `python pseudo_label.py --input gado.webm --output data/pseudo`.
5. Avaliação nos vídeos (mesmo protocolo p/ os 3 modelos):
   - det/frame e s/frame via `tune.py` (cache por modelo).
   - confirmados / IDs / frag via `tune.py --matchings 0.90 --tolerances 0`.
   - erro de contagem onde houver `ground_truth.json`.
   - Tabela Etapa 4 em `docs/PLANO-TREINAMENTO.md` + `results.png`.
6. Decisão Etapa 5: só troca `MODEL_WEIGHTS` se ganhar em mAP E erro; senão mantém `yolo11ft`.
7. Atualizar `APRESENTACAO.md` por último com faixa de resultados + ressalvas (sem número único).

## Fora de escopo (regras do repo)

- Zero rotulagem manual: sem desenhar caixas.
- Sem commitar `*.pt`, vídeos, `runs/`, `cache/`, chave de API.
- Sem refatorar contagem/tracker sem debate (regra 5).
- Parâmetros novos só em `config.py`.
