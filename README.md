# 🛰️ Sistema de Detecção, Rastreamento e Contagem Aérea de Bovinos

Aplicação em Python de alta performance para monitoramento e contagem automatizada de rebanho bovino através de filmagens aéreas de drones.

---

## 👥 Decisões da Squad de Engenharia

O comitê técnico multidisciplinar estruturou a aplicação para o hardware **AMD Ryzen 7 5700U** em modo **offline**, considerando que o drone realiza curvas e giros de 360°:

1. **Imunidade a Giros 360° (Agente 3 - QA):**
   * Em vez de depender exclusivamente de uma linha estática na tela (que causaria contagens repetidas a cada volta do drone sobre o pasto), o sistema conta com um **Totalizador de IDs Únicos com Filtro de Persistência** (`MIN_FRAMES_PERSISTENCE`).
   * Um animal só é registrado no totalizador se for rastreado continuamente por pelo menos 15 frames, filtrando falsos positivos efêmeros (copas de árvores, pedras, sombras).
   * O HUD também exibe a métrica **NO QUADRO**, indicando quantos bovinos estão visíveis naquele exato segundo de filmagem.

2. **Otimização Ryzen 7 5700U (Agente 2 - Performance):**
   * Configuração de multithreading do PyTorch (`torch.set_num_threads(8)`) para paralelização nos 8 núcleos da CPU Zen 2.
   * Utilização do modelo YOLOv8n (nano) para inferência ultra-rápida sem GPU dedicada.
   * Uso de estruturas vetorizadas do `supervision` para cálculo de IoU e associação do ByteTrack.

3. **Arquitetura Modular Desacoplada (Agente 1 - Lead Architect):**
   * `config.py`: Centraliza todos os hiperparâmetros, diretórios e limiares de confiança.
   * `detector.py`: Encapsula a inferência do YOLO e converte predições em objetos `sv.Detections`.
   * `tracker.py`: Gerencia o algoritmo ByteTrack com predição de Filtro de Kalman e buffer de oclusão.
   * `counter.py`: Calcula a contagem ativa, o acúmulo de IDs únicos e suporte opcional a `LineZone`.
   * `main.py`: Executa o pipeline de vídeo, renderiza o HUD superior profissional e salva o vídeo anotado.

---

## 🚀 Instalação das Dependências

Abra o terminal (Prompt de Comando ou PowerShell) na pasta do projeto e instale as dependências:

```bash
pip install -r requirements.txt
```

> **Dica para CPU:** Para garantir a versão do PyTorch mais leve e otimizada para CPU:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install ultralytics supervision opencv-python tqdm
> ```

---

## 🎬 Como Executar

### 1. Execução Básica (Padrão)
Coloque o vídeo gravado do drone com o nome `input_video.mp4` dentro da pasta do projeto e execute:

```bash
python main.py
```

O vídeo final anotado será salvo como `output_annotated_video.mp4`.

---

### 2. Execução com Argumentos Customizados
Você pode especificar o caminho de entrada, saída e modelo através da linha de comando:

```bash
python main.py --input "C:/caminho/do/meu_video.mp4" --output "C:/caminho/resultado.mp4" --model yolov8n.pt
```

---

### 3. Ativar LineZone (Para voos retilíneos em corredores/porteiras)
Se você tiver um trecho de vídeo onde o voo é puramente retilíneo ou o gado passa por uma porteira/corredor, adicione a flag:

```bash
python main.py --enable-linezone
```

---

## 📊 Elementos Gráficos no Vídeo Anotado

* **Bounding Box Azul Celeste:** Delimitação nítida do bovino detectado.
* **Rastro Laranja (Trace):** Histórico visual do deslocamento do animal nos últimos frames.
* **ID & Confiança:** Identificador único do animal persistido pelo ByteTrack (`#ID Confiança`).
* **Painel Superior (HUD):**
  * **NO QUADRO:** Animais ativos no frame atual.
  * **TOTAL CONFIRMADO:** Contagem cumulativa validada pelo filtro de persistência.
  * **Frame / FPS:** Telemetria de desempenho em tempo real do processador Ryzen.
