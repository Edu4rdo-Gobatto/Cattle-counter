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
| **1. Leitura** | Abre o vídeo e lê **um quadro por vez** (não carrega o vídeo inteiro na memória) | [main.py:133](main.py#L133) abre · [main.py:205](main.py#L205) lê cada quadro |
| **2. Identificação** | A YOLO analisa o quadro e devolve as caixas onde há animais, com a confiança e a classe | [detector.py:54-65](detector.py#L54-L65) |
| **3. Rastreamento** | O ByteTrack liga as caixas do quadro atual às do anterior e dá um **número fixo (ID)** a cada animal | [tracker.py:42](tracker.py#L42) |
| **4. Contagem** | Conta quantos quadros cada ID apareceu. Com **15 quadros** ou mais, o animal entra no total confirmado | [counter.py:77-86](counter.py#L77-L86) |
| **5. Desenho e gravação** | Desenha caixas, IDs, rastros e o painel (HUD) e grava o quadro no vídeo de saída | [main.py:226-269](main.py#L226-L269) · HUD em [main.py:26](main.py#L26) |
| **6. Exibição ao vivo** | Mostra o quadro anotado numa janela (espaço pausa, F tela cheia, Q sai) | [main.py:272-291](main.py#L272-L291) |
| **Orquestração** | O laço principal chama as etapas 1 a 6 para cada quadro | [main.py:202-291](main.py#L202-L291) |

### 2.1 Leitura ([main.py](main.py))
* `cv2.VideoCapture` abre o arquivo e lê largura, altura, FPS e total de quadros ([main.py:133-140](main.py#L133-L140)).
* Cada `cap.read()` traz **um** quadro (uma imagem como matriz NumPy). Isso permite processar vídeos de qualquer tamanho sem estourar a RAM.

### 2.2 Identificação ([detector.py](detector.py))
* **YOLO** ("You Only Look Once") é uma rede neural que olha a imagem inteira **uma vez** e devolve todos os objetos de uma só vez. Por isso é rápida.
* Usamos o modelo **`yolo11ft.pt` (YOLOv11)**, já treinado por terceiros em imagens aéreas de gado (*transfer learning* — nenhuma rotulagem manual). Ele tem uma única classe: `cow`.
* Filtros aplicados ([config.py:36-44](config.py#L36-L44)):
  * **Confiança ≥ 0,25:** descarta palpites fracos sem perder animais distantes.
  * **IoU 0,60 (NMS):** quando duas caixas se sobrepõem demais sobre o mesmo animal, fica só a melhor — o valor alto preserva animais agrupados.
  * **Classe de interesse:** só passa a classe `cow`.
* **SAHI opcional (`--slicing`):** fatia o quadro em janelas 640×640 com sobreposição e junta as caixas. Detecta melhor animais pequenos/distantes, mas é ~10× mais lento ([detector.py:49-58](detector.py#L49-L58)).
* A saída é convertida para `sv.Detections`, o formato da biblioteca *supervision* ([detector.py:73](detector.py#L73)).

### 2.3 Rastreamento ([tracker.py](tracker.py))
* A YOLO sozinha não sabe que o boi do quadro 10 é o mesmo do quadro 11. O **ByteTrack** resolve isso.
* O **filtro de Kalman** prevê onde cada animal estará no próximo quadro, com base na posição e na velocidade. Depois o ByteTrack associa as novas caixas às previsões.
* Diferencial do ByteTrack: ele **também usa as detecções de baixa confiança** para não perder um animal que ficou parcialmente encoberto.
* Se o animal some (sombra, árvore), o ID fica guardado por **45 quadros** (~1,5 s) antes de ser descartado ([config.py:53](config.py#L53)).

### 2.4 Contagem ([counter.py](counter.py))
O sistema mostra duas métricas no HUD:
* **NO QUADRO:** quantos animais estão visíveis agora.
* **TOTAL CONFIRMADO:** quantos IDs diferentes já apareceram por pelo menos **15 quadros consecutivos** ([config.py:87](config.py#L87)). A tolerância a falhas curtas de detecção (`STREAK_GAP_TOLERANCE`) existe, mas está em 0 — sem contagem manual de referência não dá para provar que tolerar falhas melhora o resultado.

O filtro de persistência:
* **descarta ruído**: uma pedra detectada por 2 quadros nunca chega a 15;
* **evita contar duas vezes**: o total é um *conjunto* de IDs, então o mesmo ID nunca é somado de novo, mesmo que o drone gire 360°.

A **LineZone** (linha virtual que conta quem cruza) existe, mas está **desligada** ([config.py:66](config.py#L66)). Ela só funciona bem com o drone voando em linha reta sobre um corredor.

---

## 3. Como rodar

**Demo (um comando só):**
```powershell
.\run.bat                     # Windows: usa input_video.mp4, video-gado.mp4 ou gado-teste.mp4
.\run.bat outro-video.mp4     # ou um vídeo específico
./run.sh                      # Linux
```
Abre a janela com o tracking ao vivo. Controles: **[ESPAÇO]** pausa, **[F]** tela cheia, **[Q]/[ESC]** encerra. O vídeo anotado também é salvo em `output_annotated_video.mp4`.

**Passo a passo manual:**
```powershell
.venv\Scripts\activate                                                # 1. ativa o ambiente Python
python -m unittest test_pipeline -v                                   # 2. testes automáticos (9 testes)
python main.py --input video-gado.mp4 --output resultado.mp4 --show   # 3. processa e exibe
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
| `video-gado.mp4` / `video-gado-anotado.mp4` | Vídeo real de teste e o resultado já processado |

---

## 4. Resultados

| Teste | Resultado |
| :--- | :--- |
| Testes automáticos | **9 de 9 passaram** |
| Modelo especializado `yolo11ft.pt` (YOLOv11, transfer learning) | ~8x mais detecções que o `yolov8n` no vídeo denso |
| `video-gado.mp4` (20 s, 611 quadros, Nelore, drone baixo e inclinado) | **66 confirmados** (eram 72 com o COCO) para **70 contados à mão** = **erro −6%**; ~4,5 quadros/s |
| `gado.webm` (1080x1920, rebanho denso, drone alto) | **132 confirmados** (eram 23 com o COCO) |
| Ajuste distantes/agrupados (`conf 0,25`, `iou 0,60`) | +8–11% na contagem, mesma velocidade |
| Calibração do rastreador (`matching` 0,70 → 0,90) | fragmentação **2,61 → 1,75** e **3,30 → 1,42**; IDs 180 → 103 e 439 → 182 |
| `conf 0,25 → 0,15` (18/11) | **56 → 67 confirmados** com os mesmos 89 IDs: erro vs. humano **−20% → −4%** |
| Filtro de marca d'água (18/11) | remove 1 falso positivo do logo do TikTok; total final **66 (−6%)** |
| SAHI (`--slicing`) no `gado.webm` | 18,9 → 20,9 detecções/frame, ~1,9 s/frame |
| Limitação restante | recall do detector: as caixas veem ~50% dos animais de um quadro (22 de 40); fragmentação 1,35 |

---

## 5. Perguntas prováveis do professor

**"Vocês treinaram o modelo?"**
Não, de propósito. Usamos *transfer learning*: um modelo já treinado em milhares de imagens. Treinar do zero com poucas imagens daria *overfitting* (o modelo decora em vez de aprender). O nosso trabalho de engenharia está no rastreamento e na contagem.

**"Por que não contar as caixas de cada quadro e somar?"**
Porque o mesmo boi aparece em centenas de quadros. Em 20 s de vídeo são 611 quadros, e somar contaria cada animal centenas de vezes. O rastreamento dá um ID único por animal e contamos IDs, não caixas.

**"Por que 15 quadros?"**
Equivale a meio segundo a 30 FPS. É curto o bastante para não perder um boi que aparece rapidamente e longo o bastante para descartar detecções falsas de 1 ou 2 quadros. A persistência é **consecutiva**: se o ID some, a contagem recomeça. É ajustável em [config.py:80](config.py#L80).

**"O que acontece se um boi passar debaixo de uma árvore?"**
O ByteTrack mantém o ID guardado por 45 quadros (~1,5 s). Se o boi reaparecer nesse tempo, ele continua com o mesmo número.

**"Por que o modelo genérico confundia o gado?"**
O COCO identifica os Nelore (gado branco) como `sheep` e, em voo alto, quase não via os animais (2,3 detecções/frame no vídeo denso). Trocamos por um modelo treinado em gado aéreo (`yolo11ft.pt`), que só tem a classe `cow` e detecta ~8× mais.

**"A contagem está certa?"**
Agora temos como responder: **erro de −6%**. Contamos à mão 3 quadros (40, 44 e 35 bois) e estimamos **≈70 animais distintos** no vídeo. O sistema conta **66**. Para chegar lá mudamos duas coisas medidas: baixamos o limiar de confiança de 0,25 para 0,15 (de 56 para 67 confirmados, sem criar nenhum ID novo — as caixas fracas só mantêm o rastreio vivo) e filtramos um falso positivo sistemático, a marca d'água do TikTok, que era detectada como boi e ficava parada na tela (1 px em 339 quadros). Antes dessas mudanças o erro era **−20%**. O que ainda limita é o **recall**: as caixas pegam ~50% dos animais de um quadro, então parte do acerto vem de o tracker recuperar o animal em quadros seguintes.

**"Como vocês sabem que a contagem está certa?"**
Com contagem humana de referência (`validation/ground_truth.json`): 3 quadros contados à mão e o total de animais distintos do vídeo. O `tune.py` agora reporta o **erro %** contra esse total, não só a fragmentação. Também comparamos os animais de duas configurações **pela posição** (não pelo número do ID, que é renumerado a cada execução): os 56 animais do limiar 0,25 estão todos entre os 66 atuais.

**"Roda em tempo real?"**
Na CPU do notebook, a ~5 FPS com o modelo especializado (o vídeo tem 30 FPS): processa em cerca de 6× o tempo do vídeo. O código já aceita GPU (CUDA), basta trocar `DEVICE` em [config.py:21](config.py#L21), e com GPU passaria do tempo real. Para contar um rebanho, processar depois do voo é suficiente.

**"Como vocês sabem que a fragmentação diminuiu?"**
Criamos um medidor (`validate.py report`): ele roda o rastreamento e conta, para cada ID, quando apareceu, quando sumiu, o maior intervalo sem ser visto e o maior trecho contínuo. Com isso sai o número de IDs por animal confirmado — a fragmentação. Também medimos o **pico simultâneo** (máximo de animais no quadro ao mesmo tempo), que é um limite inferior do rebanho. Antes: 3,30 IDs por animal no vídeo denso; depois do ajuste: 1,42. Um script separado (`tune.py`) reaplica o rastreamento em combinações de parâmetros usando as detecções já salvas em cache, então cada teste leva segundos em vez de minutos.

**"O que é IoU / NMS?"**
IoU (*Intersection over Union*) mede quanto duas caixas se sobrepõem. O NMS (*Non-Maximum Suppression*) usa o IoU para apagar caixas duplicadas sobre o mesmo animal e deixar só a de maior confiança.

**"Como vocês garantem que o código funciona?"**
Com testes automáticos em [test_pipeline.py](test_pipeline.py): persistência (só conta após 15 quadros), reset da persistência quando o ID some, ruído (2 quadros não conta), giro de 360° (3 animais por 50 quadros = 3, não 150), objeto fixo na tela (marca d'água sai do total), HUD e SAHI. Hoje são **9 testes, todos passando**.

**"Por que separar em vários arquivos?"**
Cada arquivo tem uma responsabilidade. Para trocar o modelo, mexemos só no detector. Para trocar a regra de contagem, só no counter. Isso facilita testar, manter e evoluir o código.

**"Quais as limitações?"**
* Troca de ID quando a câmera se move muito, o que infla o total.
* Animais muito pequenos/distantes ainda podem passar; o SAHI ajuda, mas é ~10x mais lento.
* Animais muito juntos ainda podem virar uma caixa só (mitigado com `iou` maior).
* Os modelos prontos foram treinados em nadir; no vídeo oblíquo o ganho é menor.

**"Próximos passos?"**
* ~~Usar um modelo especializado em gado aéreo (Roboflow).~~ **Adotado:** `models/yolo11ft.pt` (YOLOv11, `cattle_graze_model`).
* **Fontes encontradas (prontas para download):** Roboflow Universe → "aerial cows" (mAP@50 86.2%, P 88.8%, R 81.3%); ICAERUS-EU/UC3_Livestock_Monitoring → cow YOLOv8 (nadir, ~100 m); Hugging Face → `rmayormartins/yolov5-farm-cattle`, `xxilytoo/cattle_graze_model`, `cpraschl/bambi-models`. Testar via `--model` nos vídeos `video-gado.mp4` e `gado.webm`.
* Validar a contagem contra uma contagem manual.
* Compensar o movimento da câmera.
* Usar GPU.
* Adaptar o HUD para vídeo vertical.

---

## 6. Checklist antes da aula

- [ ] Rodar uma vez antes e confirmar o total (**66**):
  `.\.venv\Scripts\python.exe main.py --input video-gado.mp4 --output resultado.mp4 --show --end-frame 491`
  O `.\run.bat` (demo ao vivo) não corta a vinheta, então conta alguns bois a mais — use-o para mostrar o tracking, não para conferir o número.
- [ ] Ter o [video-gado-anotado.mp4](video-gado-anotado.mp4) aberto como **plano B**, caso o computador da sala seja lento ou não tenha o ambiente.
- [ ] No Windows, use `.\run.bat`. O `./run.sh` é só para Linux.
- [ ] O primeiro quadro pode demorar alguns segundos (carregamento do modelo). Avise a turma para não parecer travado.
- [ ] A janela ao vivo roda a ~8–11 FPS na CPU, mais devagar que o vídeo original. Isso é normal. Se perguntarem, veja "Roda em tempo real?".
