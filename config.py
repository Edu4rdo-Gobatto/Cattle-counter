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

# 8 núcleos físicos dedicados à inferência PyTorch
CPU_NUM_THREADS = 8

# Caminho ou nome do modelo YOLO
# Pode ser 'yolov8n.pt', 'yolo11n.pt' ou arquivo customizado treinado para visão aérea (ex: 'best.pt')
MODEL_WEIGHTS = "yolov8n.pt"

# Resolução de inferência (640 para alta velocidade; 1280 para pequenos bovinos a grande altitude)
INFERENCE_IMAGE_SIZE = 640

# Limiares de detecção
CONFIDENCE_THRESHOLD = 0.35  # Limiar de confiança mínima para considerar um animal
IOU_THRESHOLD = 0.45         # Limiar de Non-Maximum Suppression (NMS)

# No modelo COCO padrão do YOLO:
# Classe 19 = 'cow' (bovino)
# Se estiver usando um modelo customizado (onde classe 0 é gado), mude para [0] ou None para todas.
CLASSES_OF_INTEREST = [19]

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
