# Tarefas para fechar o projeto

Base: validação de 05/11/2026 (`validation/RELATORIO-VALIDACAO.md`). Em cada tarefa, **Critério** define quando ela está pronta.

Legenda de prioridade:
- **P0:** bloqueia a credibilidade da apresentação;
- **P1:** melhora a contagem;
- **P2:** manutenção;
- **P3:** acabamento.

## Feito em 05/11/2026

- [x] Validação completa do commit `ba494f5`: reprodução exata dos números publicados.
- [x] `--end-frame` / `config.END_FRAME`: corta a vinheta do TikTok. `video-gado.mp4 --end-frame 491` dá **56 confirmados / 89 IDs / frag 1,59** (antes 59 / 103 / 1,75, com ≥3 falsos positivos no logo).
- [x] `tune.py`: a coluna `IDs` agora é impressa (antes as colunas saíam deslocadas).
- [x] `.gitignore`: removida a linha solta `aa`.
- [x] Limpeza: vídeos gerados, `cache/`, `__pycache__/` e `saida.mp4` apagados; mantidos `video-gado.mp4` e `gado.webm`.

## Feito em 18/11/2026

- [x] **T1. Referência humana.** Contados os 3 frames (`15 → 40`, `98 → 44`, `149 → 35`) e estimado o total distinto (**≈70**, 69–70) assistindo até o frame 491. Gravado em `validation/ground_truth.json` (`--total 70 --end 491`).
- [x] **T4. Métrica de acerto.** `validate.py gt --total/--end/--note`; `load_ground_truth(kind="total")`; `tune.py` usa o total distinto quando a execução cobre a mesma faixa (senão avisa) e ganhou `--end-frame`.
- [x] **Filtro de marca d'água** (não estava nas tarefas): o watermark do TikTok confirmava como animal. `counter.static_track_ids()` descarta ID com centro variando ≤ 10 px em ≥ 100 frames. `conf 0.25 → 0.15` reduziu o erro de **−20% para −6%** (66 confirmados para 70).

## Feito em 27/11/2026 (sem o dono, só o alcançável sem GPU/robô humano)

- [x] **T2 parcial — Etapa 0 do plano.** Criados `train.py` (wrapper `YOLO.train` com tudo vindo de `config.TRAIN_*`), `pseudo_label.py` (opção B, conf ≥ 0,5, separado por vídeo, só para treino), `config.py §7` (`TRAIN_*`, `PSEUDO_*`) e `notebooks/treino_colab.ipynb`. Falta: baixar dataset A, linha de base `yolo val`, treinos A e A+B no Colab e tabela da Etapa 4.
- [x] **T3 parcial — docs.** `APRESENTACAO.md §2` atualizada para `conf 0,15`, refs de linha removidas (estavam quebradas), filtro de objeto fixo + `--end-frame 491` documentados, checklist com fallback para `gado.webm`. `README.md` sem "squad/agentes" e com `CPU_NUM_THREADS`. Falta: decisão final A vs A+B na apresentação.
- [x] **Demo sem `video-gado.mp4`.** `run.bat`/`run.sh` agora aceitam `gado.webm` como reserva (o `.mp4` não está na pasta).
- [x] **`.gitignore`.** Adicionado `runs/` (pesos gerados do treino).

## P0: credibilidade

- [x] **T1. Referência humana (você).**
  - Contar os bovinos em `validation/video-gado_f00015.png`, `f00098.png` e `f00149.png`, ignorando as caixas. Registrar com `python validate.py gt --video video-gado.mp4 --frame <N> --count <C>`.
  - Se possível, estimar também o **total de animais distintos** do vídeo, assistindo devagar.
  - *Critério:* `validation/ground_truth.json` existe e foi feito por humano. — **Feito.**
- [ ] **T2. Plano de treinamento mínimo** (`docs/PLANO-TREINAMENTO.md`): `train.py`, `pseudo_label.py`, dataset público anotado, treino no Colab e tabela antes × depois.
  - *Critério:* tabela da Etapa 4 preenchida e `results.png` salvo.
- [ ] **T3. Atualizar `APRESENTACAO.md` e `README.md`** com os números sem vinheta (56 / 89 / 1,59) e as ressalvas honestas:
  - a referência ainda está pendente;
  - fragmentação menor não é o mesmo que contagem certa.
  - *Critério:* nenhum número dos slides diverge dos logs em `validation/`. — **README e CLAUDE.md feitos; `APRESENTACAO.md` pendente.**
- [x] **T4. Métrica de acerto em `validate.py`.**
  - Hoje `load_ground_truth()` usa o `max()` entre frames (o pico). Adicionar uma referência de **total distinto por vídeo** (chave separada no JSON) e fazer o `tune.py` reportar o erro contra ela.
  - Mexe na lógica de validação; combinar antes.
  - *Critério:* `tune.py` mostra o erro % contra o total distinto. — **Feito.**

## P1: qualidade da contagem

- [ ] **T5. Bloquear saltos de track.** Com `min-match 0.90` (IoU > 0,10), o #47 saltou de um boi para o logo.
  - Opção barata: teto de deslocamento por frame (proporcional ao tamanho da caixa) antes de aceitar a associação.
  - *Critério:* nenhum ID troca de objeto nas 6 inspeções e a frag não sobe mais que 0,1.
- [ ] **T6. Compensação de movimento da câmera (CMC).**
  - Homografia ORB + RANSAC entre frames consecutivos, aplicada às caixas preditas antes do matching.
  - Ataca a causa raiz da fragmentação em voo e pode permitir voltar a um `min-match` mais estrito (0.70–0.80), sem fusões.
  - *Critério:* menor erro vs. T1 do que o 0.90 atual.
  - Depende de T1.
- [ ] **T7. Recall em rebanho denso.** As caixas englobam 2–6 animais e muitos ficam sem caixa.
  - Testar `imgsz 960` e SAHI **contra a referência humana**, não contra detecções/frame. O fine-tune de T2 também ataca isso.
  - *Critério:* detectados/frame mais perto da contagem humana dos 3 frames.
- [ ] **T8. Escolher `STREAK_GAP_TOLERANCE` por erro**, não por fragmentação (a tolerância derruba a frag por construção).
  - Depende de T1 e T4.

## P2: manutenção

- [ ] **T9. Migrar o tracker** antes da `supervision` 0.31 (o `sv.ByteTrack` será removido): pacote `trackers`, ou o ByteTrack/BoT-SORT do Ultralytics. BoT-SORT já tem CMC embutido e pode resolver T6 junto.
  - *Critério:* mesma reprodução de 56 / 89 ± 2 com o novo tracker, ou diferença explicada.
- [ ] **T10. Testes.**
  - Teste do `--end-frame`.
  - Teste de borda com > 15 frames: o caso de 10 frames é trivial.
  - Teste de regressão do `tune.evaluate()` com detecções sintéticas fixas.
- [ ] **T11. `gado.webm` declara 645 frames e decodifica 644.** Usar os frames lidos no resumo e no nome do cache.
- [ ] **T12. Device automático:** `cuda` → `mps` → `cpu` no `config.py` (hoje fixo em `cpu`).

## P3: acabamento

- [ ] **T13. Migração para `src/`** (arquitetura alvo do `CLAUDE.md` §5). Combinar antes; preservar a lógica de persistência.
- [ ] **T14. README:** remover o texto de "squad/agentes" e citar `CPU_NUM_THREADS` (metade de `os.cpu_count()`) em vez de `set_num_threads(8)`.
- [ ] **T15. Recorte automático de vinheta**, opcional: detectar frames sem pasto (brilho/verde médio) em vez de exigir `--end-frame` manual.

## Ordem sugerida

T1 → T3 → T2 (em paralelo com T9) → T4 → T5/T6 → T7/T8 → P2/P3.
