# Roteiro de Apresentação — Contador Aéreo de Bovinos

> **Demo em um comando (Windows/PowerShell):** `.\run.bat`. Ele abre a janela com o vídeo e o tracking ao vivo. No Linux, use `./run.sh`.

---

## 1. Pitch de 30 segundos

> "Contar gado no pasto é manual, demorado e sujeito a erro. Nosso sistema recebe um vídeo gravado por drone e conta os bovinos automaticamente. Usamos uma rede neural (YOLO) para **identificar** os animais em cada imagem, um algoritmo de rastreamento (ByteTrack) para **reconhecer o mesmo animal** de um quadro para o outro, e uma regra de contagem que só confirma um animal depois que ele aparece por meio segundo. Assim não contamos o mesmo boi duas vezes nem confundimos sombra com boi. A saída é um vídeo com cada animal marcado e o total na tela."

---

## 2. Como funciona, etapa por etapa

```plaintext
 vídeo .mp4 ─► 1. LEITURA ─► 2. IDENTIFICAÇÃO ─► 3. RASTREAMENTO ─► 4. CONTAGEM ─► 5. DESENHO ─► vídeo anotado
              (OpenCV)       (YOLO)              (ByteTrack)         (persistência)  (HUD)
```

| Etapa | O que acontece | Onde está no código |
| :--- | :--- | :--- |
| **0. Configuração** | Todos os parâmetros num lugar só: modelo, confiança, classes, tempo de persistência | [config.py](config.py) |
| **1. Leitura** | Abre o vídeo e lê **um quadro por vez** (não carrega o vídeo inteiro na memória) | [main.py](main.py) abre com `VideoCapture` e lê cada quadro com `cap.read()` no laço principal |
| **2. Identificação** | A YOLO analisa o quadro e devolve as caixas onde há animais, com a confiança e a classe | [detector.py](detector.py) (`_predict` + `detect`) |
| **3. Rastreamento** | O ByteTrack liga as caixas do quadro atual às do anterior e dá um **número fixo (ID)** a cada animal | [tracker.py](tracker.py) (`update_with_detections`) |
| **4. Contagem** | Conta quantos quadros cada ID apareceu. Com **15 quadros** ou mais, o animal entra no total confirmado | [counter.py](counter.py) (`update`: streak consecutivo + `confirmed_unique_ids`) |
| **5. Desenho e gravação** | Desenha caixas, IDs, rastros e o painel (HUD) e grava o quadro no vídeo de saída | [main.py](main.py) (`draw_hud` + `VideoWriter`) · HUD em `draw_hud` |
| **6. Exibição ao vivo** | Mostra o quadro anotado numa janela (espaço pausa, F tela cheia, Q sai) | [main.py](main.py) (bloco `show_window`) |
| **Orquestração** | O laço principal chama as etapas 1 a 6 para cada quadro | [main.py](main.py) (`process_video`) |

### 2.1 Leitura ([main.py](main.py))
* `cv2.VideoCapture` abre o arquivo e lê largura, altura, FPS e total de quadros.
* Cada `cap.read()` traz **um** quadro (uma imagem como matriz NumPy). Isso permite processar vídeos de qualquer tamanho sem estourar a RAM.
* `--end-frame` permite cortar vinhetas e créditos. O corte no frame 491 foi usado apenas no `video-gado.mp4`, que não está mais disponível.

### 2.2 Identificação ([detector.py](detector.py))
* **YOLO** ("You Only Look Once") é uma rede neural que olha a imagem inteira **uma vez** e devolve todos os objetos de uma só vez. Por isso é rápida.
* Usamos o modelo **`yolo11ft.pt` (YOLOv11)**, já treinado por terceiros em imagens aéreas de gado (*transfer learning* — nenhuma rotulagem manual). Ele tem uma única classe: `cow`.
* Filtros aplicados ([config.py](config.py)):
  * **Confiança ≥ 0,15:** calibrado em 18/11 no vídeo oblíquo histórico, que não está mais disponível. Caixas fracas (0,15–0,25) não criam IDs novos (a ativação é 0,30), mas mantêm o track vivo quando o animal pisca por 1–2 quadros.
  * **IoU 0,60 (NMS):** quando duas caixas se sobrepõem demais sobre o mesmo animal, fica só a melhor — o valor alto preserva animais agrupados.
  * **Classe de interesse:** só passa a classe `cow`.
* **SAHI opcional (`--slicing`):** fatia o quadro em janelas 640×640 com sobreposição e junta as caixas. Detecta melhor animais pequenos/distantes, mas é ~10× mais lento ([detector.py](detector.py)).
* A saída é convertida para `sv.Detections`, o formato da biblioteca *supervision*.

### 2.3 Rastreamento ([tracker.py](tracker.py))
* A YOLO sozinha não sabe que o boi do quadro 10 é o mesmo do quadro 11. O **ByteTrack** resolve isso.
* O **filtro de Kalman** prevê onde cada animal estará no próximo quadro, com base na posição e na velocidade. Depois o ByteTrack associa as novas caixas às previsões.
* Diferencial do ByteTrack: ele **também usa as detecções de baixa confiança** para não perder um animal que ficou parcialmente encoberto. Por isso o `conf 0,15` ajuda mesmo com ativação em 0,30.
* Se o animal some (sombra, árvore), o ID fica guardado por **45 quadros** (~1,5 s) antes de ser descartado ([config.py](config.py)).
* Calibração de 04/11: `MINIMUM_MATCHING_THRESHOLD = 0,90` (aceita IoU > 0,10 na associação). Reduziu a fragmentação de 2,61 para 1,75 no oblíquo e de 3,30 para 1,42 no denso.

### 2.4 Contagem ([counter.py](counter.py))
O sistema mostra duas métricas no HUD:
* **NO QUADRO:** quantos animais estão visíveis agora.
* **TOTAL CONFIRMADO:** quantos IDs diferentes já apareceram por pelo menos **15 quadros consecutivos** ([config.py](config.py)). A tolerância a falhas curtas de detecção (`STREAK_GAP_TOLERANCE`) permanece em 0; ainda não foi comparada contra a referência humana do `gado.webm`.

O filtro de persistência:
* **descarta ruído**: uma pedra detectada por 2 quadros nunca chega a 15;
* **evita contar duas vezes**: o total é um *conjunto* de IDs, então o mesmo ID nunca é somado de novo, mesmo que o drone gire 360°.
* **descarta objeto fixo na tela**: marca d'água/logo com centro variando ≤ 10 px em ≥ 100 frames sai do total final (`static_track_ids`). No vídeo oblíquo histórico, o filtro removeu um falso positivo do TikTok.

A **LineZone** (linha virtual que conta quem cruza) existe, mas está **desligada** ([config.py](config.py)). Ela só funciona bem com o drone voando em linha reta sobre um corredor.

---

## 3. Como rodar

**Demo (um comando só):**
```powershell
.\run.bat                     # Windows: usa o primeiro vídeo disponível, incluindo gado.webm
.\run.bat outro-video.mp4     # ou um vídeo específico
./run.sh                      # Linux
```
Abre a janela com o tracking ao vivo. Controles: **[ESPAÇO]** pausa, **[F]** tela cheia, **[Q]/[ESC]** encerra. O vídeo anotado também é salvo em `output_annotated_video.mp4`.

**Passo a passo manual:**
```powershell
.venv\Scripts\activate                                                # 1. ativa o ambiente Python
python -m unittest test_pipeline -v                                   # 2. testes automáticos (10 testes)
python main.py --input gado.webm --output resultado.mp4 --show        # 3. processa e exibe
```

* Durante a execução aparece uma barra de progresso. No fim, o terminal mostra os quadros processados, o tempo, o FPS e o **total de bovinos**.
* Opções: `--show` abre a janela ao vivo; `--model outro.pt` troca o modelo; `--enable-linezone` liga a linha de contagem.

### Arquivos do projeto
| Arquivo | Para que serve |
| :--- | :--- |
| [main.py](main.py) | Ponto de entrada: laço principal, HUD e gravação |
| [config.py](config.py) | Todos os parâmetros ajustáveis |
| [detector.py](detector.py) | Identificação com YOLO |
| [tracker.py](tracker.py) | Rastreamento com ByteTrack |
| [counter.py](counter.py) | Regras de contagem |
| [test_pipeline.py](test_pipeline.py) | Testes automáticos da lógica |
| [validate.py](validate.py) | Mede a fragmentação por ID e gera imagens para contagem manual |
| [tune.py](tune.py) | Compara parâmetros do ByteTrack usando detecções cacheadas |
| [generate_test_video.py](generate_test_video.py) | Gera um vídeo sintético para teste |
| [run.bat](run.bat) / [run.sh](run.sh) | Atalho da demo (Windows / Linux) |
| [requirements.txt](requirements.txt) | Bibliotecas necessárias |
| `gado.webm` | Vídeo de desenvolvimento, 1080×1920, 644 quadros decodificados |

---

## 4. Resultados

| Teste | Resultado |
| :--- | :--- |
| Testes automáticos | **10 de 10 passaram** |
| Modelo especializado `yolo11ft.pt` (YOLOv11, transfer learning) | ~8x mais detecções que o `yolov8n` no vídeo denso |
| `gado.webm` com `yolo11ft.pt`, conf 0,15, matching 0,90 | **129 confirmados / 184 IDs / frag 1,43**, contra 100 distintos na referência humana: erro **+29%**; 19,7 detecções/frame |
| Fine-tune A (`best_a.pt`) no teste público | P 0,760 / R 0,607 / mAP50 0,652; no `gado.webm`, 3,0 detecções/frame e 0 confirmados no matching 0,90 |
| Decisão do modelo | `best_a.pt` não foi adotado: ganhou no teste público, mas teve erro −100% no vídeo; `yolo11ft.pt` permanece padrão |
| `video-gado.mp4` | Resultados históricos, não reproduzíveis neste checkout porque o arquivo não está disponível |
| Ajuste distantes/agrupados (`conf 0,25`, `iou 0,60`) | +8–11% na contagem, mesma velocidade |
| Calibração histórica do rastreador (`matching` 0,70 → 0,90) | fragmentação caiu nos vídeos de teste da época; o vídeo oblíquo não está mais disponível |
| SAHI (`--slicing`) no `gado.webm` | 18,9 → 20,9 detecções/frame, ~1,9 s/frame |
| Limitação restante | o erro de contagem varia por domínio; o fine-tune A no dataset público não transferiu para o vídeo denso |

---

## 5. Perguntas prováveis do professor

**"Vocês treinaram o modelo?"**
Sim. Fizemos fine-tuning do YOLOv11 em um dataset público de gado aéreo já anotado, sem desenhar caixas. No teste independente desse dataset, o modelo A obteve P 0,760, R 0,607 e mAP50 0,652, contra mAP50 0,061 da base. No vídeo de desenvolvimento, porém, A caiu para 3,0 detecções por frame e não confirmou nenhum ID no protocolo padrão. Por isso não substituímos o modelo adotado. Treinar do zero com poucas imagens também aumentaria o risco de *overfitting*.

**"Por que não contar as caixas de cada quadro e somar?"**
Porque o mesmo boi aparece em centenas de quadros. Em 20 s de vídeo são 611 quadros, e somar contaria cada animal centenas de vezes. O rastreamento dá um ID único por animal e contamos IDs, não caixas.

**"Por que 15 quadros?"**
Equivale a meio segundo a 30 FPS. É curto o bastante para não perder um boi que aparece rapidamente e longo o bastante para descartar detecções falsas de 1 ou 2 quadros. A persistência é **consecutiva**: se o ID some, a contagem recomeça. É ajustável em [config.py:80](config.py#L80).

**"O que acontece se um boi passar debaixo de uma árvore?"**
O ByteTrack mantém o ID guardado por 45 quadros (~1,5 s). Se o boi reaparecer nesse tempo, ele continua com o mesmo número.

**"Por que o modelo genérico confundia o gado?"**
O COCO identifica os Nelore (gado branco) como `sheep` e, em voo alto, quase não via os animais (2,3 detecções/frame no vídeo denso). Trocamos por um modelo treinado em gado aéreo (`yolo11ft.pt`), que só tem a classe `cow` e detecta ~8× mais.

**"A contagem está certa?"**
No vídeo de desenvolvimento `gado.webm`, a referência humana estima 100 animais distintos. A configuração padrão confirma 129 IDs, erro de **+29%**. Nos três quadros de referência, as contagens humanas são 38, 12 e 47; o detector base encontrou 29, 13 e 42 caixas. Caixa não equivale necessariamente a um animal, pois pode cobrir mais de um bovino ou conter duplicatas. O fine-tune A não resolveu esse domínio: confirmou zero IDs no matching 0,90. Portanto o resultado é uma estimativa, não uma contagem exata.

**"Como vocês sabem que a contagem está certa?"**
Usamos a referência humana em `validation/ground_truth.json`: três contagens de quadro e uma estimativa de 100 animais distintos no `gado.webm`. O modelo padrão erra +29% nesse vídeo. Essa referência é específica ao vídeo de desenvolvimento e não prova a precisão em outros voos. A avaliação no split test público mede detecção por caixa, não a contagem de animais únicos no vídeo.

**"Roda em tempo real?"**
Na CPU do notebook, a ~5 FPS com o modelo especializado (o vídeo tem 30 FPS): processa em cerca de 6× o tempo do vídeo. O código já aceita GPU (CUDA), basta trocar `DEVICE` em [config.py:21](config.py#L21), e com GPU passaria do tempo real. Para contar um rebanho, processar depois do voo é suficiente.

**"Como vocês sabem que a fragmentação diminuiu?"**
Criamos um medidor (`validate.py report`): ele roda o rastreamento e conta, para cada ID, quando apareceu, quando sumiu, o maior intervalo sem ser visto e o maior trecho contínuo. Com isso sai o número de IDs por animal confirmado — a fragmentação. Também medimos o **pico simultâneo** (máximo de animais no quadro ao mesmo tempo), que é um limite inferior do rebanho. Antes: 3,30 IDs por animal no vídeo denso; depois do ajuste: 1,42. Um script separado (`tune.py`) reaplica o rastreamento em combinações de parâmetros usando as detecções já salvas em cache, então cada teste leva segundos em vez de minutos.

**"O que é IoU / NMS?"**
IoU (*Intersection over Union*) mede quanto duas caixas se sobrepõem. O NMS (*Non-Maximum Suppression*) usa o IoU para apagar caixas duplicadas sobre o mesmo animal e deixar só a de maior confiança.

**"Como vocês garantem que o código funciona?"**
Com testes automáticos em [test_pipeline.py](test_pipeline.py): persistência (só conta após 15 quadros), reset da persistência quando o ID some, ruído (2 quadros não conta), giro de 360° (3 animais por 50 quadros = 3, não 150), objeto fixo na tela (marca d'água sai do total), HUD, SAHI e isolamento do cache por modelo. Hoje são **10 testes, todos passando**.

**"Por que separar em vários arquivos?"**
Cada arquivo tem uma responsabilidade. Para trocar o modelo, mexemos só no detector. Para trocar a regra de contagem, só no counter. Isso facilita testar, manter e evoluir o código.

**"Quais as limitações?"**
* Troca de ID quando a câmera se move muito, o que infla o total.
* Animais muito pequenos/distantes ainda podem passar; o SAHI ajuda, mas é ~10x mais lento.
* Animais muito juntos ainda podem virar uma caixa só (mitigado com `iou` maior).
* Os modelos prontos foram treinados em nadir; no vídeo oblíquo o ganho é menor.

**"Próximos passos?"**
* Retomar e avaliar A+B, que combina o dataset público com pseudo-rótulos automáticos de `gado.webm`. A tentativa local foi interrompida por limite de tempo na primeira época.
* Medir outros vídeos com uma referência humana antes de generalizar o erro.
* Testar compensação de movimento da câmera para reduzir fragmentação em voos altos.

---

## 6. Checklist antes da aula

- [ ] Rodar a demo com `gado.webm` e conferir a saída, sem tratar 129 como verdade universal:
  `.\.venv\Scripts\python.exe main.py --input gado.webm --output resultado.mp4 --show`
  O `video-gado.mp4` não está mais disponível. `gado.webm` é vídeo de desenvolvimento e tem referência própria em `validation/ground_truth.json`.
- [ ] Ter um vídeo anotado aberto como **plano B**, caso o computador da sala seja lento ou não tenha o ambiente.
- [ ] No Windows, use `.\run.bat`. O `./run.sh` é só para Linux.
- [ ] O primeiro quadro pode demorar alguns segundos (carregamento do modelo). Avise a turma para não parecer travado.
- [ ] A janela ao vivo roda a ~4–8 FPS na CPU, mais devagar que o vídeo original. Isso é normal. Se perguntarem, veja "Roda em tempo real?".
