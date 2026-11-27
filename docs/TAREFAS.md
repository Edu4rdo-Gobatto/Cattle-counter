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

## Feito em 27/11/2026

- [x] **T2, fine-tune A.** Modelo `best_a.pt` validado localmente no split test: P 0,760 / R 0,607 / mAP50 0,652 / mAP50-95 0,282. No vídeo de desenvolvimento, 3,0 detecções/frame e 0 confirmados em matching 0,90. Não foi adotado.
- [x] **T2, referência do `gado.webm`.** Frames 18→38, 67→12, 147→47; total distinto estimado 100 em [0:644]. Registrado em `validation/ground_truth.json`.
- [x] **T2, pseudo-rótulos A+B gerados.** 22 frames amostrados, 362 caixas em 20 frames não vazios.
- [x] **T3.** `APRESENTACAO.md` e `README.md` atualizados com o fine-tune A, a queda no vídeo de desenvolvimento e a decisão de manter `yolo11ft.pt`.
- [x] **Demo sem `video-gado.mp4`.** `run.bat`/`run.sh` agora aceitam `gado.webm` como reserva (o `.mp4` não está na pasta).
- [x] **`.gitignore`.** Adicionado `runs/` (pesos gerados do treino).

## Feito em 27/11/2026 (2ª parte — sem vídeo oficial, execução autônoma)

- [x] **Validação `gado.webm [0:150]` (sem humano ainda).** `report`: 12,9/frame, pico 31, 41 confirmados / 66 IDs / frag 1,61. Grid: 0.70 → 3,94; 0.80 → 2,12; **0.90 → 1,61** (confirma o padrão). Cache em `cache/gado_f0_150_c0.15_i0.60_s640.npz`, CSV em `validation/tune_gado_f0_150_c0.15.csv`.
- [x] **Frames p/ referência humana do `gado.webm`.** Gerados `validation/gado_f00018.png`, `f00067.png`, `f00147.png`; contagens e total distinto registrados no `ground_truth.json`.
- [x] **Dataset sem API key.** `Francesco/aerial-cows` (CC BY 4.0, 1084/299/340) baixado e convertido para YOLO.
- [x] **Linha base e fine-tune A.** Avaliados no split test público. A teve mAP50 maior, mas erro −100% no vídeo, então o modelo padrão não mudou.
- [ ] **A+B e gráficos do treino A.** A tentativa local parou após 15/70 batches da primeira época por limite de tempo; retomar no Colab ou em execução com várias horas. O `results.png` de A não veio do Colab; as curvas PR e matriz de confusão da validação local estão em `runs/detect/val-3/`.

## P0: credibilidade

- [x] **T1. Referência humana.** `validation/ground_truth.json` contém as contagens do `gado.webm` (frames 18, 67 e 147) e o total distinto estimado 100 em [0:644]. O vídeo oblíquo antigo não está mais disponível.
- [ ] **T2. Plano de treinamento mínimo** (`docs/PLANO-TREINAMENTO.md`): A+B ainda precisa terminar e ser avaliado; falta recuperar `results.png` do treino A, se disponível.
- [x] **T3. Atualizar apresentação e README.** Resultados atuais, limitações do teste de transferência e decisão da Etapa 5 documentados.
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
