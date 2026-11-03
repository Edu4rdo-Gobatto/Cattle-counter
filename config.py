"""
Módulo de Configuração Central (config.py)
Projeto: Sistema de Contagem Aérea de Bovinos
Otimizado para: AMD Ryzen 7 5700U (8C/16T, CPU Processing)
"""

from pathlib import Path
import os

# ==============================================================================
# 1. DIRETÓRIOS E ARQUIVOS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
INPUT_VIDEO_PATH = BASE_DIR / "input_video.mp4"
OUTPUT_VIDEO_PATH = BASE_DIR / "output_annotated_video.mp4"

# ==============================================================================
# 2. HARDWARE & INFERÊNCIA (AMD RYZEN 7 5700U)
# ==============================================================================
# 'cpu' para execução nativa no Ryzen 7
DEVICE = "cpu"

# Núcleos dedicados à inferência PyTorch (metade dos threads lógicos ~ núcleos físicos)
CPU_NUM_THREADS = max(1, (os.cpu_count() or 2) // 2)

# Caminho ou nome do modelo YOLO
# Modelo especializado em gado aéreo (transfer learning, pesos prontos da internet):
# https://github.com/xxilytoo/cattle_graze_model (YOLOv11 fine-tuned, classe 0 = cow)
# Benchmark local: ~8x mais detecções que o yolov8n no vídeo denso (gado.webm).
MODEL_WEIGHTS = str(BASE_DIR / "models" / "yolo11ft.pt")

# Alternativa genérica (COCO): "yolov8n.pt" — nesse caso use CLASSES_OF_INTEREST = [18, 19]

# Resolução de inferência (640 para alta velocidade; 1280 para pequenos bovinos a grande altitude)
INFERENCE_IMAGE_SIZE = 640

# Limiares de detecção
# conf 0.25 + iou 0.60: calibrado nos dois vídeos para pegar animais distantes e
# preservar caixas sobrepostas de bovinos agrupados (mesma velocidade, +8-11% na contagem).
CONFIDENCE_THRESHOLD = 0.25  # Limiar de confiança mínima para considerar um animal
IOU_THRESHOLD = 0.60         # IoU do NMS (maior preserva animais agrupados)

# O modelo especializado usa apenas a classe 0 = 'cow'.
# Para o modelo COCO (yolov8n.pt), troque por [18, 19] (sheep + cow).
CLASSES_OF_INTEREST = [0]

# ==============================================================================
# 2.1 DETECÇÃO POR FATIAS (SAHI / INFERENCE SLICER)
# ==============================================================================
# Divide cada frame em fatias sobrepostas e detecta em cada uma. Melhora a
# detecção de animais pequenos/distantes em voos altos, ao custo de ~8x mais
# inferências. Desligado por padrão para manter a demo ao vivo rápida.
USE_SLICING = False

# Lado de cada fatia, em pixels (igual ao INFERENCE_IMAGE_SIZE)
SLICE_SIZE = 640

# Sobreposição entre fatias, em pixels (~20% de 640 reduz animais cortados na borda)
SLICE_OVERLAP = 128

# ==============================================================================
# 3. RASTREAMENTO ESPACIAL (BYTETRACK VIA SUPERVISION)
# ==============================================================================
# Taxa de quadros assumida caso não seja detectada no metadado do vídeo
DEFAULT_FPS = 30

# Limiar de confiança para transformar uma detecção em um track ativo
TRACK_ACTIVATION_THRESHOLD = 0.30

# Buffer de retenção para oclusões (copas de árvores, sombras ou perdas temporárias)
# 45 frames a 30 FPS = 1.5 segundos de tolerância à oclusão
LOST_TRACK_BUFFER = 45

# Limiar de IoU mínimo para associar detecções a tracks existentes no filtro de Kalman
MINIMUM_MATCHING_THRESHOLD = 0.70

# ==============================================================================
# 4. LÓGICA DE CONTAGEM & ROBUSTEZ A GIROS 360° E CURVAS
# ==============================================================================
# Quantidade mínima de frames consecutivos que um ID precisa persistir
# antes de ser confirmado como um animal real (elimina falsos positivos de vegetação/sombras)
MIN_FRAMES_PERSISTENCE = 15

# Modo LineZone (Opcional - útil quando o voo é puramente retilíneo em funis/porteiras)
ENABLE_LINEZONE = False

# Coordenadas relativas da LineZone na tela (percentual da largura e altura: 0.0 a 1.0)
# (x_start, y_start) e (x_end, y_end)
LINEZONE_START_RELATIVE = (0.05, 0.50)  # 5% da borda esquerda, 50% da altura
LINEZONE_END_RELATIVE = (0.95, 0.50)    # 95% da borda direita, 50% da altura

# ==============================================================================
# 5. ESTÉTICA VISUAL E TELEMETRIA (HUD NO VÍDEO ANOTADO)
# ==============================================================================
# Espessura da linha do Bounding Box
BOX_THICKNESS = 2

# Exibir rastro de movimento dos animais (em frames)
TRACE_LENGTH = 30
TRACE_THICKNESS = 2

# Cores em formato BGR para OpenCV
COLOR_PRIMARY_HUD = (34, 139, 34)     # Verde Floresta
COLOR_ACCENT = (0, 165, 255)          # Laranja Âmbar
COLOR_BOX = (255, 191, 0)             # Deep Sky Blue
COLOR_TEXT = (255, 255, 255)          # Branco
COLOR_PANEL_BG = (20, 20, 20)         # Cinza escuro quase preto
PANEL_OPACITY = 0.75                  # Transparência do HUD superior

# ==============================================================================
# 6. EXIBIÇÃO AO VIVO (MODO APRESENTAÇÃO)
# ==============================================================================
# Abre uma janela do OpenCV mostrando o vídeo anotado enquanto ele é processado
SHOW_WINDOW = False
WINDOW_NAME = "Contador Aereo de Bovinos"

# Altura inicial da janela em pixels (a janela pode ser redimensionada ou maximizada)
WINDOW_HEIGHT = 900
