# Plano de Treinamento Mínimo (fine-tuning sem rotulagem manual)

**Objetivo acadêmico:** mostrar ao professor que o projeto domina o ciclo completo de visão computacional, com dados, treino, avaliação e comparação, sem desenhar uma única caixa à mão (regra 1 do `CLAUDE.md`).

**Entrega mínima que sustenta a nota:** uma tabela "antes × depois" com o mesmo protocolo de avaliação, os gráficos do Ultralytics (`results.png`, curva PR, matriz de confusão) e um parágrafo explicando o porquê do resultado, **mesmo que o ganho seja pequeno ou nulo**. O que reprova é não medir.

---

## 1. Estratégia

Faremos **fine-tuning (transfer learning)** do `models/yolo11ft.pt`, que já detecta gado aéreo, em vez de treinar do zero.

| Opção | Dados | Rotulagem | Papel no projeto |
|---|---|---|---|
| **A (obrigatória)** | Dataset público **já anotado** em formato YOLO (aerial cows/cattle) | Nenhuma: vem pronta | Treino e **avaliação honesta** (split de teste independente) |
| **B (complemento)** | Frames dos nossos vídeos rotulados **automaticamente** pelo próprio `yolo11ft` (pseudo-rótulos, conf ≥ 0,5) | Nenhuma: gerada pelo modelo | Adaptação ao domínio (Nelore branco, câmera oblíqua) |

A opção B nunca é usada para avaliar: o modelo seria julgado pelos próprios rótulos, ou seja, por viés de confirmação. Ela só entra no treino. A avaliação no vídeo é feita pela **contagem** (número), com a referência humana da Fase 2 da validação.

### Fontes de dados (opção A), em ordem de preferência

1. **Roboflow Universe:** "aerial cows" (Roboflow 100). O `README.md` registra mAP@50 de 86,2% para o modelo de referência. Baixar no formato **YOLOv8/YOLO11** (Download Dataset → YOLOv8), que traz `train/`, `valid/`, `test/` e `data.yaml`. Requer conta gratuita e `ROBOFLOW_API_KEY` em variável de ambiente, nunca no código.
2. **ICAERUS UC3 Livestock Monitoring:** o mesmo grupo dos `cow_weight_v*.pt`. Vista nadir; útil para variedade.
3. Qualquer outro dataset do Universe com o selo de anotações prontas e licença que permita uso acadêmico (CC BY 4.0 é o comum). **Registrar licença e versão** no relatório.

Destino local: `data/datasets/<nome>/`, já ignorado pelo `.gitignore` (regra da arquitetura alvo).

---

## 2. Ambiente

| Onde | Tempo estimado | Quando usar |
|---|---|---|
| **Google Colab (GPU T4 gratuita)** | ~1–2 min/época para ~1–3 mil imagens a 640 px | **Recomendado** para o treino principal |
| CPU local (Ryzen 7 5700U) | Dezenas de minutos por época; inviável para 30+ épocas | Só para o *smoke test* (1 época, poucas imagens) e para a inferência |

As estimativas acima são ordens de grandeza, não medidas. Medir no smoke test e corrigir aqui.

---

## 3. Passo a passo

### Etapa 0: preparação (código, ~1 dia)

1. Criar `train.py` na raiz, como wrapper fino do `ultralytics.YOLO(...).train()`, com todos os parâmetros vindos do `config.py` (regra 6):
   `TRAIN_DATA_YAML`, `TRAIN_EPOCHS=30`, `TRAIN_IMGSZ=640`, `TRAIN_BATCH=16`, `TRAIN_FREEZE=10`, `TRAIN_PATIENCE=10`, `TRAIN_PROJECT="runs/train"`, `TRAIN_SEED=0`.
2. Criar `pseudo_label.py` (opção B):
   - extrai 1 frame a cada N dos vídeos;
   - roda o `CattleDetector` com conf ≥ 0,5;
   - grava `images/` + `labels/*.txt` no formato YOLO;
   - **separa por vídeo** para não vazar frames quase idênticos entre treino e validação.
3. Adicionar `runs/` e `data/` ao `.gitignore`.
4. Escrever o notebook `notebooks/treino_colab.ipynb` com as células: instalar requirements → baixar dataset → `train.py` → `val` → baixar `best.pt`.

### Etapa 1: linha de base (antes de treinar)

```bash
yolo val model=models/yolo11ft.pt data=data/datasets/<nome>/data.yaml split=test imgsz=640
```

Anotar P, R, mAP@50 e mAP@50-95 do `yolo11ft` **no split de teste do dataset público**. Esta é a linha "antes".

### Etapa 2: treino (opção A)

```bash
python train.py   # yolo11ft.pt -> runs/train/a/weights/best.pt
```

- `freeze=10` congela o backbone. Isso é transfer learning de fato: menos épocas, menos overfitting, argumento forte para a banca.
- `patience=10` para o treino se o mAP de validação estagnar.
- Seed fixa para o resultado ser reproduzível.

### Etapa 3: treino (opção A + B, ablação)

Mesmo comando, com o `data.yaml` combinando o dataset público e os pseudo-rótulos. Gera `runs/train/ab/`.

```bash
python pseudo_label.py --input gado.webm --output data/pseudo
python train.py --name ab --pseudodata data/aerial-cows-ab.yaml
```

### Etapa 4: avaliação (mesmo protocolo para os 3 modelos)

| Métrica | Onde | Ferramenta |
|---|---|---|
| P / R / mAP@50 / mAP@50-95 | split **test** do dataset público | `yolo val` |
| Detecções/frame | `gado.webm [0:644]`, vídeo de desenvolvimento | `tune.py --model <pesos>` |
| Confirmados / IDs / frag | `gado.webm [0:644]`, `matching=0.90`, `tol=0` | `tune.py --model <pesos>` |
| **Erro de contagem** | mesma faixa da referência humana, total 100 | `validation/ground_truth.json` |

O `tune.py` e o `validate.py` aceitam `--model`; caches, relatórios e imagens recebem o nome do modelo para não misturar execuções. O cache sem sufixo continua reservado ao modelo padrão.

`video-gado.mp4` não está mais disponível. A comparação de vídeo abaixo usa apenas o `gado.webm` e o mesmo limiar de detecção (conf 0,15, IoU 0,60, imgsz 640).

| Modelo | P (teste) | R (teste) | mAP@50 | mAP@50-95 | det/frame | Confirmados / IDs / frag | Erro vs. 100 |
|---|---|---|---|---|---|---|---|
| `yolo11ft` (base) | 0,216 | 0,143 | 0,061 | 0,019 | 19,7 | 129 / 184 / 1,43 | +29% |
| Fine-tune A (`best_a.pt`) | 0,760 | 0,607 | 0,652 | 0,282 | 3,0 | 0 / 12 / não definido | −100% |
| Fine-tune A+B (`best_ab.pt`) | pendente | pendente | pendente | pendente | pendente | pendente | pendente |

O `best_a.pt` foi validado no split test localmente. Na avaliação de vídeo, o pico foi 31 caixas em um frame, mas o tracker chegou a apenas 3 ativos simultâneos; no matching 0,90, nenhum ID confirmou e 12 IDs foram criados. O grid de matching produziu 0 confirmados / 16 IDs em 0,70, 1 / 15 em 0,80 e 0 / 12 em 0,90. A fragmentação de A em 0,90 não é definida porque não houve IDs confirmados. A avaliação está registrada em `validation/tune_gado_f0_644_c0.15_mbest_a.csv` e usa o cache `cache/gado_f0_644_c0.15_i0.60_s640_mbest_a.npz`.

O aumento de mAP no dataset público não se transferiu para o vídeo de desenvolvimento. O fine-tune A não substitui o modelo padrão.

Para A+B, foram gerados pseudo-rótulos de 22 frames de `gado.webm` com `PSEUDO_CONF=0.5`: 362 caixas em 20 frames não vazios. O treino local foi interrompido pelo limite de execução após 15/70 batches da primeira época; nessa CPU, 30 épocas exigiriam algumas horas. Não há `best_ab.pt` nem métricas de A+B. Retomar no Colab ou em uma execução com tempo suficiente. Os pseudo-rótulos são usados apenas no treino; a avaliação continua no split test público e contra a referência humana do vídeo.

Crédito do dataset: "aerial-cows from Roboflow 100 (Roboflow Universe), CC BY 4.0."

As curvas PR e a matriz de confusão da validação local de A estão em `runs/detect/val-3/` e são regeneráveis. Os gráficos do treino A (`results.png`) não vieram junto com o checkpoint do Colab.

### Etapa 5: decisão

- A decisão já está tomada para A: manter `MODEL_WEIGHTS=models/yolo11ft.pt`. A ganhou no mAP do dataset público, mas teve erro de −100% no vídeo de desenvolvimento.
- Reavaliar a decisão quando A+B terminar. Só trocar o modelo padrão se a mesma variante melhorar tanto o mAP no split test quanto o erro de contagem no vídeo.
- O resultado negativo de A é um achado válido: desempenho no dataset público não garante transferência para gado denso filmado de ângulo oblíquo.

---

## 4. Cronograma sugerido

| Dia | Atividade | Entregável |
|---|---|---|
| 1 | `train.py`, `pseudo_label.py`, `config.py` (TRAIN_*), smoke test 1 época em CPU | Script rodando |
| 2 | Baixar dataset A, linha de base `yolo val`, contagem humana dos 3 frames | Linha "antes" + `ground_truth.json` |
| 3 | Treinos A e A+B no Colab | `best.pt` ×2 + `results.png` |
| 4 | Avaliação nos vídeos e tabela final | Tabela da Etapa 4 |
| 5 | Slides: "por que fine-tuning e não do zero", gráficos, limitações | Seção nova na `APRESENTACAO.md` |

---

## 5. Riscos e como responder ao professor

| Pergunta provável | Resposta apoiada em dados |
|---|---|
| "Vocês treinaram algo?" | Sim: fine-tuning do YOLOv11 com backbone congelado, em dataset público anotado, com mAP antes/depois no mesmo split de teste. |
| "Por que não treinar do zero?" | Poucas imagens + domínio específico = overfitting. O transfer learning reaproveita features genéricas. Mostrar a curva de validação. |
| "Pseudo-rótulo não é trapaça?" | Só entra no treino. A avaliação é em dados anotados por terceiros e em contagem humana. Mostrar a ablação A × A+B. |
| "E se piorou?" | Resultado negativo medido é resultado. Mostrar onde o gargalo realmente está (fragmentação/saltos de ID). |
| Licença dos dados | Registrar fonte, versão e licença do dataset no relatório. |

**Fora do escopo (proibido pela regra 1):** desenhar ou corrigir caixas à mão, ou montar um dataset próprio anotado manualmente.
