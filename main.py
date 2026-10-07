"""
Pipeline Principal (main.py)
Orquestra o fluxo de processamento de vídeo offline:
Decodificação -> Detecção YOLO -> Rastreamento ByteTrack -> Contagem -> HUD & Anotação -> Gravação
"""

import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

import config
from detector import CattleDetector
from tracker import CattleTracker
from counter import CattleCounter


def draw_hud(
    frame: np.ndarray,
    counts: dict,
    current_frame: int,
    total_frames: int,
    current_fps: float,
    enable_linezone: bool,
) -> np.ndarray:
    """
    Desenha um painel superior de telemetria (HUD) translúcido e profissional no frame.
    """
    height, width = frame.shape[:2]
    hud_height = 85

    # Criação da camada de overlay translúcida
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (width, hud_height),
        config.COLOR_PANEL_BG,
        -1,
    )
    # Linha divisória inferior do HUD
    cv2.line(
        overlay,
        (0, hud_height),
        (width, hud_height),
        config.COLOR_ACCENT,
        2,
    )
    # Aplica transparência
    cv2.addWeighted(
        overlay,
        config.PANEL_OPACITY,
        frame,
        1.0 - config.PANEL_OPACITY,
        0,
        frame,
    )

    # Tipografia e Métricas
    font = cv2.FONT_HERSHEY_DUPLEX
    
    # 1. Título do Sistema
    cv2.putText(
        frame,
        "MONITORAMENTO AEREO DE BOVINOS",
        (20, 28),
        font,
        0.65,
        config.COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )

    # 2. Métrica: Animais Ativos no Frame Atual (Imune a giros 360°)
    active_str = f"NO QUADRO: {counts['active_in_frame']}"
    cv2.putText(
        frame,
        active_str,
        (20, 65),
        font,
        0.85,
        config.COLOR_PRIMARY_HUD,
        2,
        cv2.LINE_AA,
    )

    # 3. Métrica: Total Acumulado Confirmado (Filtro de persistência)
    total_str = f"TOTAL CONFIRMADO: {counts['confirmed_unique_total']}"
    cv2.putText(
        frame,
        total_str,
        (280, 65),
        font,
        0.85,
        config.COLOR_ACCENT,
        2,
        cv2.LINE_AA,
    )

    # 4. Métrica: LineZone (caso ativada)
    if enable_linezone:
        line_str = f"LINE IN: {counts['line_in']} | OUT: {counts['line_out']}"
        cv2.putText(
            frame,
            line_str,
            (620, 65),
            font,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # 5. Telemetria Técnica: Frame e FPS de Processamento
    progress_str = f"Frame: {current_frame}/{total_frames} | {current_fps:.1f} FPS"
    text_size = cv2.getTextSize(progress_str, font, 0.55, 1)[0]
    cv2.putText(
        frame,
        progress_str,
        (width - text_size[0] - 20, 48),
        font,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return frame


def process_video(
    input_path: Path,
    output_path: Path,
    model_path: str = config.MODEL_WEIGHTS,
    enable_linezone: bool = config.ENABLE_LINEZONE,
):
    """Executa o pipeline completo de contagem aérea e gravação do vídeo anotado."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"\n[ERRO CRÍTICO] Vídeo de entrada não encontrado em:\n{input_path}\n"
            f"Por favor, coloque o vídeo no diretório do projeto ou especifique via argumento --input."
        )

    print(f"\n{'='*70}")
    print("INICIANDO PIPELINE DE CONTAGEM AEREA DE BOVINOS (AMD RYZEN 7 5700U)")
    print(f"{'='*70}")
    print(f"Vídeo de Entrada: {input_path}")
    print(f"Vídeo de Saída:   {output_path}")
    print(f"Modelo:           {model_path}")
    print(f"Modo LineZone:    {'ATIVADO' if enable_linezone else 'DESATIVADO (Rastreamento por Persistência)'}")

    # Leitura dos metadados do vídeo com OpenCV
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise IOError(f"Falha ao abrir o arquivo de vídeo: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolução:        {width}x{height} @ {fps:.2f} FPS")
    print(f"Total de Frames:  {total_frames}")

    # Configuração do escritor de vídeo de saída (Codec MP4V compatível universalmente)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Inicialização dos módulos do pipeline
    detector = CattleDetector(model_path=model_path)
    tracker = CattleTracker(frame_rate=int(fps))
    counter = CattleCounter(
        frame_width=width,
        frame_height=height,
        enable_linezone=enable_linezone,
    )

    # Anotadores Visuais da biblioteca Supervision
    box_annotator = sv.BoxAnnotator(
        thickness=config.BOX_THICKNESS,
        color=sv.Color.from_hex("#00BFFF"),  # Azul celeste vibrante
    )
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_padding=3,
        text_color=sv.Color.from_hex("#FFFFFF"),
        color=sv.Color.from_hex("#00BFFF"),
    )
    trace_annotator = sv.TraceAnnotator(
        trace_length=config.TRACE_LENGTH,
        thickness=config.TRACE_THICKNESS,
        color=sv.Color.from_hex("#FF8C00"),  # Rastro laranja visível contra grama verde
    )

    # Loop principal de processamento
    frame_idx = 0
    start_total_time = time.time()
    last_time = start_total_time
    proc_fps = 0.0

    try:
        with tqdm(total=total_frames, desc="Processando Vídeo", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                t_now = time.time()
                elapsed_frame = t_now - last_time
                last_time = t_now
                if elapsed_frame > 0:
                    proc_fps = 1.0 / elapsed_frame

                # 1. Inferência YOLO
                detections = detector.detect(frame)

                # 2. Atualização do ByteTrack
                tracked_detections = tracker.update(detections)

                # 3. Lógica de Contagem
                counts = counter.update(tracked_detections)

                # 4. Anotações Gráficas no Frame
                annotated_frame = frame.copy()

                # Desenha rastro dos animais
                annotated_frame = trace_annotator.annotate(
                    scene=annotated_frame,
                    detections=tracked_detections,
                )

                # Desenha bounding boxes
                annotated_frame = box_annotator.annotate(
                    scene=annotated_frame,
                    detections=tracked_detections,
                )

                # Rótulos (ID do boi + Confiança)
                labels = []
                if tracked_detections.tracker_id is not None and len(tracked_detections.tracker_id) > 0:
                    for idx, tid in enumerate(tracked_detections.tracker_id):
                        conf_val = ""
                        if tracked_detections.confidence is not None and len(tracked_detections.confidence) > idx:
                            conf_val = f" {tracked_detections.confidence[idx]:.2f}"
                        labels.append(f"#{tid}{conf_val}")

                    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame,
                        detections=tracked_detections,
                        labels=labels,
                    )

                # Desenha LineZone se habilitada teste
                annotated_frame = counter.annotate_linezone(annotated_frame)

                # Desenha Painel HUD Superior
                annotated_frame = draw_hud(
                    frame=annotated_frame,
                    counts=counts,
                    current_frame=frame_idx,
                    total_frames=total_frames,
                    current_fps=proc_fps,
                    enable_linezone=enable_linezone,
                )

                # Grava o quadro anotado no arquivo de saída
                writer.write(annotated_frame)
                pbar.update(1)
    finally:
        # Liberação garantida de recursos para evitar corrupção de arquivo de vídeo
        cap.release()
        writer.release()
    total_time = time.time() - start_total_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0

    print(f"\n{'='*70}")
    print("PROCESSAMENTO CONCLUIDO COM SUCESSO!")
    print(f"{'='*70}")
    print(f"Frames Processados:       {frame_idx}")
    print(f"Tempo Total:              {total_time:.2f} s")
    print(f"Velocidade Média da CPU:  {avg_fps:.2f} FPS")
    print(f"Bovinos Únicos Validados: {counts['confirmed_unique_total']}")
    if enable_linezone:
        print(f"LineZone In / Out:        {counts['line_in']} / {counts['line_out']}")
    print(f"Vídeo Anotado Salvo Em:   {output_path.resolve()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sistema de Detecção, Rastreamento e Contagem Aérea de Bovinos."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(config.INPUT_VIDEO_PATH),
        help="Caminho do arquivo de vídeo de entrada.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.OUTPUT_VIDEO_PATH),
        help="Caminho para salvar o vídeo anotado de saída.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.MODEL_WEIGHTS,
        help="Pesos do modelo YOLO (.pt).",
    )
    parser.add_argument(
        "--enable-linezone",
        action="store_true",
        default=config.ENABLE_LINEZONE,
        help="Habilita a contagem por LineZone além do totalizador de persistência.",
    )

    args = parser.parse_args()

    process_video(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model_path=args.model,
        enable_linezone=args.enable_linezone,
    )
