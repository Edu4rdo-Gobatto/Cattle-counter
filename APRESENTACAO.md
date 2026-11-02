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
* Usamos o modelo `yolov8n.pt` ("n" = nano, o menor), pré-treinado no dataset **COCO**, que tem 80 classes, entre elas `cow` (19) e `sheep` (18).
* Filtros aplicados ([config.py:34-40](config.py#L34-L40)):
  * **Confiança ≥ 0,35:** descarta palpites fracos.
  * **IoU 0,45 (NMS):** quando duas caixas se sobrepõem demais sobre o mesmo animal, fica só a melhor.
  * **Classes de interesse:** só passam as classes de animal configuradas.
* A saída é convertida para `sv.Detections`, o formato da biblioteca *supervision* ([detector.py:65](detector.py#L65)).

### 2.3 Rastreamento ([tracker.py](tracker.py))
* A YOLO sozinha não sabe que o boi do quadro 10 é o mesmo do quadro 11. O **ByteTrack** resolve isso.
* O **filtro de Kalman** prevê onde cada animal estará no próximo quadro, com base na posição e na velocidade. Depois o ByteTrack associa as novas caixas às previsões.
* Diferencial do ByteTrack: ele **também usa as detecções de baixa confiança** para não perder um animal que ficou parcialmente encoberto.
* Se o animal some (sombra, árvore), o ID fica guardado por **45 quadros** (~1,5 s) antes de ser descartado ([config.py:53](config.py#L53)).

### 2.4 Contagem ([counter.py](counter.py))
O sistema mostra duas métricas no HUD:
* **NO QUADRO:** quantos animais estão visíveis agora.
* **TOTAL CONFIRMADO:** quantos IDs diferentes já apareceram por pelo menos **15 quadros** ([config.py:63](config.py#L63)).

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
python -m unittest test_pipeline -v                                   # 2. testes automáticos (5 testes)
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
| [generate_test_video.py](generate_test_video.py) | Gera um vídeo sintético para teste |
| [run.bat](run.bat) / [run.sh](run.sh) | Atalho da demo (Windows / Linux) |
| [requirements.txt](requirements.txt) | Bibliotecas necessárias |
| `video-gado.mp4` / `video-gado-anotado.mp4` | Vídeo real de teste e o resultado já processado |

---

## 4. Resultados

| Teste | Resultado |
| :--- | :--- |
| Testes automáticos | 5 de 5 passaram |
| `video-gado.mp4` (20 s, 611 quadros, gado Nelore, drone baixo e inclinado) | **72 bovinos confirmados** em ~55 s (~11 quadros/s, só CPU) |
| Bois próximos à câmera | Detectados com confiança de 0,5 a 0,9 |
| Bois distantes no fundo | Muitos não são detectados (ficam pequenos demais) |

---

## 5. Perguntas prováveis do professor

**"Vocês treinaram o modelo?"**
Não, de propósito. Usamos *transfer learning*: um modelo já treinado em milhares de imagens. Treinar do zero com poucas imagens daria *overfitting* (o modelo decora em vez de aprender). O nosso trabalho de engenharia está no rastreamento e na contagem.

**"Por que não contar as caixas de cada quadro e somar?"**
Porque o mesmo boi aparece em centenas de quadros. Em 20 s de vídeo são 611 quadros, e somar contaria cada animal centenas de vezes. O rastreamento dá um ID único por animal e contamos IDs, não caixas.

**"Por que 15 quadros?"**
Equivale a meio segundo a 30 FPS. É curto o bastante para não perder um boi que aparece rapidamente e longo o bastante para descartar detecções falsas de 1 ou 2 quadros. É ajustável em [config.py:63](config.py#L63).

**"O que acontece se um boi passar debaixo de uma árvore?"**
O ByteTrack mantém o ID guardado por 45 quadros (~1,5 s). Se o boi reaparecer nesse tempo, ele continua com o mesmo número.

**"Por que vocês aceitam a classe ovelha?"**
Foi um achado do teste. O modelo genérico identificava os Nelore (gado branco) como `sheep`. Com só `cow`, a contagem dava 0. Como não há ovelhas no vídeo, aceitar as duas classes é seguro. A solução definitiva é um modelo especializado em gado visto de drone (Roboflow Universe), que já existe pronto.

**"O 72 está certo?"**
É uma estimativa, e o número real provavelmente é menor. A câmera do drone se move bastante. Quando o rastreador perde um animal por mais de 1,5 s, ele volta com um ID novo e é contado de novo. Para medir a precisão exata seria preciso contar à mão e comparar. Esse é o próximo passo de validação.

**"Roda em tempo real?"**
Na CPU de um notebook, roda a ~11 FPS, enquanto o vídeo tem 30 FPS. Ou seja, processa em cerca de 3× o tempo do vídeo. O código já aceita GPU (CUDA), basta trocar `DEVICE` em [config.py:21](config.py#L21), e com GPU passaria do tempo real. Para contar um rebanho, processar depois do voo é suficiente.

**"O que é IoU / NMS?"**
IoU (*Intersection over Union*) mede quanto duas caixas se sobrepõem. O NMS (*Non-Maximum Suppression*) usa o IoU para apagar caixas duplicadas sobre o mesmo animal e deixar só a de maior confiança.

**"Como vocês garantem que o código funciona?"**
Com testes automáticos em [test_pipeline.py](test_pipeline.py): persistência (só conta após 15 quadros), ruído (2 quadros não conta), giro de 360° (3 animais por 50 quadros = 3, não 150) e HUD.

**"Por que separar em vários arquivos?"**
Cada arquivo tem uma responsabilidade. Para trocar o modelo, mexemos só no detector. Para trocar a regra de contagem, só no counter. Isso facilita testar, manter e evoluir o código.

**"Quais as limitações?"**
* Troca de ID quando a câmera se move muito, o que infla o total.
* Animais distantes ou pequenos não são detectados.
* Animais muito juntos podem virar uma caixa só.
* O modelo genérico confunde raças, como o Nelore classificado como ovelha.

**"Próximos passos?"**
* Usar um modelo especializado em gado aéreo (Roboflow).
* Validar a contagem contra uma contagem manual.
* Compensar o movimento da câmera.
* Usar GPU.
* Adaptar o HUD para vídeo vertical.

---

## 6. Checklist antes da aula

- [ ] Rodar `.\run.bat` uma vez antes e confirmar o total (**72**).
- [ ] Ter o [video-gado-anotado.mp4](video-gado-anotado.mp4) aberto como **plano B**, caso o computador da sala seja lento ou não tenha o ambiente.
- [ ] No Windows, use `.\run.bat`. O `./run.sh` é só para Linux.
- [ ] O primeiro quadro pode demorar alguns segundos (carregamento do modelo). Avise a turma para não parecer travado.
- [ ] A janela ao vivo roda a ~8–11 FPS na CPU, mais devagar que o vídeo original. Isso é normal. Se perguntarem, veja "Roda em tempo real?".
