# Relatório de Validação — commit `ba494f5`

**Veredito:** APROVADO COM RESSALVAS

## Resumo executivo

- Os quatro números publicados no commit (confirmados, IDs, fragmentação) foram reproduzidos **exatamente** nos dois vídeos, via `tune.py`, `validate.py report` e `main.py`.
- Os 8 testes unitários passam. As flags `--min-match`, `--streak-tol` e `--track-act` funcionam. Os casos de borda não quebram.
- O commit mede **fragmentação**, não **acerto da contagem**. Ainda não existe referência manual confiável: a Fase 2 ficou PENDENTE_HUMANO e a Fase 3, SEM_REFERENCIA.
- **Falso positivo confirmado:** a vinheta do TikTok ocupa os frames 492–611 do `video-gado.mp4` (≈120 frames, não ~20). Pelo menos 3 dos 59 confirmados (IDs 96, 97 e 101) nasceram dentro dela, sobre o logo.
- **Associação frouxa:** com `min-match 0.90`, o ID #47, que era um boi no frame 306, termina preso ao logo. É o risco de "fusão" que o 0.90 introduz.
- **Afirmação do commit não reproduzida no default:** na janela [450:600], "0.90 → 23 confirmados" só ocorre com `streak-tol 2`. Com o default (`tol 0`) são 17. Além disso, 72% dessa janela é vinheta.
- **Recall do detector em rebanho denso:** caixas cobrem de 2 a 6 animais e vários animais ficam sem caixa. Na contagem visual do frame 15 (baixa confiança) estimei ≈39 animais para 22 detecções.

## Reprodução vs. publicado

| vídeo | min-match | métrica | esperado | obtido | status |
|---|---|---|---|---|---|
| video-gado.mp4 (611 f) | 0.70 | confirmados / IDs / frag | 69 / 180 / 2,61 | 69 / 180 / 2,61 | PASS |
| video-gado.mp4 (611 f) | 0.90 | confirmados / IDs / frag | 59 / 103 / 1,75 | 59 / 103 / 1,75 | PASS |
| gado.webm (645 f*) | 0.70 | confirmados / IDs / frag | 133 / 439 / 3,30 | 133 / 439 / 3,30 | PASS |
| gado.webm (645 f*) | 0.90 | confirmados / IDs / frag | 128 / 182 / 1,42 | 128 / 182 / 1,42 | PASS |
| video-gado [0:150] | 0.90 | confirmados / IDs / frag | 27 / 36 / 1,33 | 27 / 36 / 1,33 | PASS |
| video-gado [450:600] | — | pico | 23 | 23 | PASS |
| video-gado [450:600] | 0.70 | confirmados (tol 0) | 16 | 16 | PASS |
| video-gado [450:600] | 0.90 | confirmados | 23 | 17 (tol 0) / 23 (tol 2) | FAIL no default |

\* O `gado.webm` declara 645 frames (`CAP_PROP_FRAME_COUNT`), mas só 644 são decodificados. Os resultados acima são sobre os 644.

## Fase 0 — Ambiente e testes unitários

**Comandos:** `git rev-parse --short HEAD`, impressão de versões, `test_pipeline.py -v`, `main.py --help`.

```
ba494f5
3.11.9 | cv2 5.0.0 | supervision 0.30.6 | ultralytics 8.4.170 | numpy 2.4.6
Ran 8 tests in 18.557s — OK   (inclusive o teste de SAHI, que não foi pulado)
--help: --conf --iou --imgsz --track-act --min-match --streak-tol --slicing --show
```

**Resultado:** PASS. O `--help` não imprime os defaults. Conferi no código: `main.py` usa `config.MINIMUM_MATCHING_THRESHOLD = 0.90` e `config.STREAK_GAP_TOLERANCE = 0`.

**O que prova:** o ambiente casa com o `CLAUDE.md` e a lógica do contador passa nos testes sintéticos. Não prova nada sobre a contagem em vídeo real.

## Fase 1 — Regressão A/B

**Comandos:** `tune.py --limit 611|645 --conf 0.25 --activations 0.30 --matchings 0.70,0.90 --tolerances 0`, seguido de `validate.py report` com os defaults.

```
video-gado [0:611]  0.70 -> 69 conf / 180 IDs / 2.61 | 0.90 -> 59 conf / 103 IDs / 1.75   (YOLO 5,42 FPS)
gado.webm  [0:644]  0.70 -> 133 / 439 / 3.30         | 0.90 -> 128 / 182 / 1.42          (YOLO 5,46 FPS)
report video-gado: 103 IDs, 59 confirmados, frag 1.75, pico 23, 12 curtos, 16 fragmentos
report gado.webm:  182 IDs, 128 confirmados, frag 1.42, pico 39, 11 curtos, 27 fragmentos
```

**Resultado:** PASS. Fragmentação 1,75 ≤ 1,75 e 1,42 ≤ 1,42. Os caches foram gerados do zero nesta execução (`*_c0.25_i0.60_s640.npz`), sem reaproveitar conf diferente.

**O que prova:** a mudança 0.70 → 0.90 reduz IDs criados em 43% (`video-gado`) e 59% (`gado.webm`), de forma determinística. **Não prova** que 59 ou 128 é a contagem correta.

## Fase 2 — Contagem manual de referência

**Status:** PENDENTE_HUMANO. **Nenhum `ground_truth.json` foi escrito.**

Método tentado: recortei o frame bruto, sem caixas, em 4 faixas horizontais ampliadas 2x e com régua de pixels (`validation/fase2_recortes/`, apagado na limpeza; script em `evidencias_2026-11-05/scripts/recortes_fase2.py`).

| frame | faixa y230–340 | y330–440 | y430–560 | y550–830 | total | confiança | detectados YOLO |
|---|---|---|---|---|---|---|---|
| 15 | ≈15 | ≈14 | ≈6 | 4 | ≈39 (faixa 34–43) | baixa | 22 |
| 98 | não contado por faixa | | | | — | baixa | 20 |
| 149 | não contado por faixa | | | | — | baixa | 20 |

Por que a confiança é baixa:
- o rebanho ao fundo está muito sobreposto e tem poucos pixels por animal (576x1024);
- há animais cortados entre faixas, com risco de contagem dupla;
- o total fica muito acima de "detectados", mas pela razão inversa ao alerta do roteiro: as caixas verdes englobam 2 ou 3 animais, então o YOLO subconta.

Os frames 98 e 149 têm a mesma densidade e não foram contados por faixa.

**Para o humano contar:** `validation/video-gado_f00015.png`, `validation/video-gado_f00098.png` e `validation/video-gado_f00149.png`.

## Fase 3 — Tolerância de streak

**Status:** SEM_REFERENCIA (sem coluna de erro).

| janela | match | tol 0 | tol 1 | tol 2 | IDs | frag (tol 0→2) |
|---|---|---|---|---|---|---|
| [0:150] | 0.90 | 27 | 29 | 31 | 36 | 1,33 → 1,16 |
| [0:150] | 0.85 | 29 | 31 | 33 | 40 | 1,38 → 1,21 |
| [0:150] | 0.80 | 31 | 33 | 35 | 45 | 1,45 → 1,29 |
| [450:600] | 0.90 | 17 | 20 | 23 | 45 | 2,65 → 1,96 |
| [450:600] | 0.85 | 19 | 21 | 22 | 51 | 2,68 → 2,32 |
| [450:600] | 0.80 | 19 | 21 | 22 | 56 | 2,95 → 2,55 |

Valores em "confirmados". Pico: 20 em [0:150] e 22–23 em [450:600].

**Recomendação:** **não** mudar o default `STREAK_GAP_TOLERANCE = 0`.
- A tolerância sempre aumenta os confirmados com os IDs fixos, então a "fragmentação" cai por construção, sem nenhuma evidência de acerto.
- Em [0:150], qualquer tolerância já leva o total (29–35) acima do pico de 20 na janela.
- A janela [450:600] está 72% contaminada pela vinheta, então não serve para calibrar.

## Fase 4 — Fim a fim

| vídeo | exit | confirmados | IDs | frag | FPS | saída | bate com Fase 1 |
|---|---|---|---|---|---|---|---|
| video-gado.mp4 | 0 | 59 | 103 | 1,75 | 5,07 | 38,6 MB | sim |
| gado.webm | 0 | 128 | 182 | 1,42 | 5,92 | 92,3 MB | sim |

Os dois logs imprimem "IDs Criados pelo Tracker" e "Fragmentação". **PASS** técnico. Os MP4s foram apagados depois, a pedido do usuário (ver Anexos).

## Inspeção visual

Frames extraídos dos vídeos anotados em ~10%, 50% e 90%.

| vídeo | frame | achado | veredito |
|---|---|---|---|
| video-gado | 62 | #25 e #23 cobrem grupos de 4–6 animais; #2 cobre 2 bois; ~10 animais à esquerda e ao fundo sem caixa. Nenhum ID repetido. | SUSPEITO (recall/caixa fundida) |
| video-gado | 306 | ~14 caixas para ~30 animais visíveis (grupos à esquerda e ao fundo sem caixa). IDs distintos, sem duplicação visível. #47 está num boi. | SUSPEITO (recall) |
| video-gado | 550 | **Vinheta do TikTok:** 6 caixas sobre logo e texto (#47, #77, #93, #95, #97 e outro), "no quadro: 6". #47, que estava num boi no frame 306, agora está no logo. | **FALSO POSITIVO + fusão (a)** |
| gado.webm | 65 | #21, #27, #29 e #31 com caixas sobrepostas cobrindo 2 animais cada; ~5 animais na coluna sob a palmeira sem caixa; rastro de #4/#9 salta ~150 px para dentro da palmeira. | SUSPEITO (troca de ID) |
| gado.webm | 323 | Motion blur forte. Caixas #41, #49, #80 e #100 englobam 3–5 animais. Rastros em zigue-zague (#67, #80) compatíveis com ID pulando entre vizinhos. | SUSPEITO (fusão (a)) |
| gado.webm | 580 | #163 e #150 englobam ~5 animais. Duas caixas aninhadas em #153 e rótulos sobrepostos "#1… / #156" no mesmo lugar. ~6 animais na borda direita sem caixa. | SUSPEITO (duplicação (b)) |

Cruzamento com o CSV de IDs (`fase4_vinheta.log`): a vinheta começa no frame **492** (brilho médio < 40). Dos 59 confirmados, 6 estão vivos nela:
- **96, 97 e 101 nasceram dentro dela** (falsos positivos certos);
- 47 e 77 vinham de bois e migraram para o logo;
- 82 vai até o frame 520.

## Fase 5 — Casos de borda

| caso | exit | confirmados | IDs | status |
|---|---|---|---|---|
| `sem_gado.mp4` (30 frames pretos, 640x480) | 0 | 0 | 0 | PASS |
| `curto.mp4` (10 frames do video-gado) | 0 | 0 | 18 | PASS (trivial: 10 < persistência 15) |
| `video-gado.mp4 --min-match 0.50 --streak-tol 2 --track-act 0.25` | 0 | 68 | 925 | PASS (aceitou as flags; frag 13,60) |

O terceiro caso confirma o sentido do parâmetro: `min-match 0.50` exige IoU > 0,5 e explode a fragmentação.

## Divergências e riscos

1. **Vinheta do TikTok gera falsos positivos confirmados.** Pelo menos 3 dos 59 (IDs 96, 97, 101). A vinheta tem ~120 frames (492–611); o `CLAUDE.md` fala em ~20. Afeta também a linha de base 0.70.
2. **Associação frouxa (0.90 ⇒ IoU > 0,10)** permite que um track "salte" para outro objeto: #47 foi de um boi para o logo. É a regressão (a) prevista. A queda de fragmentação pode estar escondendo fusões.
3. **Afirmação "[450:600] 0.90 → 23" não vale no default.** Exige `tol 2`; com `tol 0` dá 17. O histórico anterior a esta validação (`logs/anterior_tune_video-gado_f450_150.csv`) já mostrava 17. A janela também é 72% vinheta.
4. **`confirmados` não é monotônico em fragmentação.** Menos fragmentação não implica contagem mais certa:
   - a tolerância sobe os confirmados com IDs fixos e derruba a "frag" por construção;
   - 0.70 → 0.90 baixa os confirmados (69 → 59) e a frag ao mesmo tempo.
   Sem referência, não dá para dizer qual está mais perto do real.
5. **Viés do `max()` em `load_ground_truth()`.** A referência é o pico entre frames, não o total de animais distintos do vídeo, o que enviesa o erro. Para um vídeo com câmera em movimento, o total real tende a ser **maior** que o pico de um frame.
6. **Recall do detector em rebanho denso.** Caixas fundidas e animais sem caixa (todas as inspeções). A subcontagem por frame e a inflação por fragmentação podem se cancelar e dar um total "plausível" pelo motivo errado.
7. **Bug de exibição no `tune.py`.** O cabeçalho tem a coluna `IDs`, mas a linha não imprime `total_ids`, então as colunas ficam deslocadas: "pico" mostra frag, "fragm" mostra pico etc. O CSV está correto. Não corrigido, conforme a regra 1.
8. **`gado.webm`: 645 declarados vs. 644 lidos.** O nome do cache usa 645 e o CSV usa 644.
9. **`sv.ByteTrack` deprecado** (FutureWarning; removido na supervision 0.31). A trava `<0.31.0` segura por enquanto.
10. **Teste `curto.mp4` é fraco:** com 10 < 15 frames, 0 confirmados é garantido. Cria 18 IDs em 10 frames.
11. **Desvios de procedimento:**
    - usei Git Bash com `timeout 900` e `PYTHONIOENCODING=utf-8`, sempre com `.venv/Scripts/python.exe`;
    - na Fase 2, recortei o frame bruto para contar;
    - os CSVs versionados que a Fase 3 sobrescreve foram restaurados do commit e os novos foram movidos para `evidencias_2026-11-05/`.
12. **Limpeza pós-validação (pedido do usuário).** Após o relatório, apagados:
    - `cache/*.npz` (4 arquivos; motivo: limpeza pedida, regenerável com `tune.py`);
    - `outputs/*.mp4` (7 vídeos), `saida.mp4`, `__pycache__/`, `validation/fase2_recortes/`;
    - os scripts e vídeos temporários em `%TEMP%\opencode`.

## Recomendações

1. **Contagem humana** dos 3 PNGs da Fase 2 (e de 2–3 frames do `gado.webm`) antes de qualquer ajuste de `min-match` ou `streak-tol`.
2. **Cortar a vinheta:** processar até o frame 491, via flag `--end-frame` ou recorte do vídeo. Isso remove ≥3 falsos positivos do total.
3. **Limitar o salto de track:** CMC (homografia entre frames) ou um teto de deslocamento por frame, antes de manter 0.90.
4. **Métrica de acerto em vez de só fragmentação:** referência de animais distintos por vídeo, não `max()` de frames.
5. **Melhorar o recall em rebanho denso:** fine-tuning do detector em um dataset aéreo **já anotado**, sem rotulagem manual.
6. Corrigir a coluna do `tune.py` e migrar o tracker antes da supervision 0.31.

## Anexos

- `git rev-parse HEAD`: `ba494f54c43b1ecc7cb2275a08ef2a979af6b055`
- Logs brutos: `validation/logs/fase0_*.log`, `fase1_*.log`, `fase2_recortes.log`, `fase3_*.log`, `fase4_*.log`, `fase5_*.log`, `fase345_exitcodes.log`
- Histórico anterior: `validation/logs/anterior_tune_video-gado_f0_150.csv`, `anterior_tune_video-gado_f450_150.csv`
- CSVs desta validação: `validation/evidencias_2026-11-05/`
  - `tune_video-gado_f0_611.csv`, `tune_gado_f0_644.csv`
  - `tune_video-gado_f0_150_fase3.csv`, `tune_video-gado_f450_150_fase3.csv`
  - `video-gado_ids_611.csv`, `gado_ids_644.csv`
- Scripts descartáveis usados: `validation/evidencias_2026-11-05/scripts/`
  - `recortes_fase2.py`, `gera_bordas.py`, `extrai_inspecao.py`, `vinheta.py`
- PNGs (ignorados pelo git, locais):
  - `validation/video-gado_f00015.png`, `f00098.png`, `f00149.png`
  - `validation/inspecao_video-gado_p10|p50|p90.png`, `validation/inspecao_gado_p10|p50|p90.png`
- Vídeos de saída e `cache/`: gerados e **apagados** na limpeza (regeneráveis pelos comandos acima).
- JSON: `validation/relatorio-validacao.json`
