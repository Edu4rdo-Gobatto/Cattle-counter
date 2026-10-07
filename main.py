"""
Pipeline Principal (main.py)
Orquestra o fluxo de processamento de vídeo offline:
Decodificação -> Detecção YOLO -> Rastreamento ByteTrack -> Contagem -> HUD & Anotação -> Gravação
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

import config
from detector import CattleDetector
from tracker import CattleTracker
from counter import CattleCounter


# Item de texto do HUD: (texto, escala da fonte, cor BGR, espessura)
HudItem = Tuple[str, float, Tuple[int, int, int], int]


def draw_hud(
    frame: np.ndarray,
    counts: Dict[str, int],
    current_frame: int,
    total_frames: int,
    current_fps: float,
    enable_linezone: bool,
) -> np.ndarray:
    """
    Desenha um painel superior de telemetria (HUD) translúcido no frame.
    O layout é calculado pela largura do quadro: em vídeos estreitos (verticais)
    as métricas quebram linha em vez de se sobreporem.
    """
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX

    # Escala proporcional à resolução, para o texto ficar legível em qualquer vídeo
    ui = min(max(min(width, height) / 720.0, 0.6), 2.0)
    margin = int(20 * ui)
    gap_x = int(30 * ui)
    gap_y = int(14 * ui)

    def item(text: str, scale: float, color: Tuple[int, int, int], thick: int) -> HudItem:
        return (text, scale * ui, color, max(1, round(thick * ui)))

    def size(it: HudItem) -> Tuple[int, int]:
        (w, h), baseline = cv2.getTextSize(it[0], font, it[1], it[3])
        return w, h + baseline

    title = item("MONITORAMENTO AEREO DE BOVINOS", 0.65, config.COLOR_TEXT, 1)
    telemetry = item(
        f"Frame: {current_frame}/{total_frames} | {current_fps:.1f} FPS", 0.55, (200, 200, 200), 1
    )
    metrics: List[HudItem] = [
        item(f"NO QUADRO: {counts['active_in_frame']}", 0.85, config.COLOR_PRIMARY_HUD, 2),
        item(f"TOTAL CONFIRMADO: {counts['confirmed_unique_total']}", 0.85, config.COLOR_ACCENT, 2),
    ]
    if enable_linezone:
        metrics.append(
            item(f"LINE IN: {counts['line_in']} | OUT: {counts['line_out']}", 0.65, (0, 255, 255), 2)
        )

    # Distribui as métricas em linhas que cabem na largura disponível
    available = width - 2 * margin
    rows: List[List[HudItem]] = [[]]
    row_width = 0
    for it in metrics:
        w = size(it)[0]
        if rows[-1] and row_width + gap_x + w > available:
            rows.append([])
            row_width = 0
        row_width += (gap_x if rows[-1] else 0) + w
        rows[-1].append(it)

    # Telemetria fica à direita do título; se não couber, vai para uma linha própria
    telemetry_inline = size(title)[0] + gap_x + size(telemetry)[0] <= available
    all_rows = [[title]] + rows + ([] if telemetry_inline else [[telemetry]])

    row_heights = [max(size(it)[1] for it in row) for row in all_rows]
    hud_height = margin + sum(row_heights) + gap_y * (len(all_rows) - 1) + margin // 2

    # Camada translúcida apenas sobre a faixa do painel (evita copiar o quadro inteiro)
    panel = frame[:hud_height]
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (width, hud_height), config.COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, config.PANEL_OPACITY, panel, 1.0 - config.PANEL_OPACITY, 0, panel)
    cv2.line(frame, (0, hud_height), (width, hud_height), config.COLOR_ACCENT, max(1, round(2 * ui)))

    # Escreve os textos linha a linha
    y = margin
    for row, row_h in zip(all_rows, row_heights):
        baseline_y = y + row_h - int(4 * ui)
        x = margin
        for it in row:
            cv2.putText(frame, it[0], (x, baseline_y), font, it[1], it[2], it[3], cv2.LINE_AA)
            x += size(it)[0] + gap_x
        if row[0] is title and telemetry_inline:
            tx = width - margin - size(telemetry)[0]
            cv2.putText(frame, telemetry[0], (tx, baseline_y), font, telemetry[1], telemetry[2], telemetry[3], cv2.LINE_AA)
        y += row_h + gap_y

    return frame


def process_video(
    input_path: Path,
    output_path: Path,
    model_path: str = config.MODEL_WEIGHTS,
    enable_linezone: bool = config.ENABLE_LINEZONE,
    show_window: bool = config.SHOW_WINDOW,
):
    """Executa o pipeline completo de contagem aérea e gravação do vídeo anotado."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"\n[ERRO CRÍTICO] Vídeo de entrada não encontrado em:\n{input_path}\n"
            f"Por favor, coloque o vídeo no diretório do projeto ou especifique via argumento --input."
        )

    print(f"\n{'='*70}")
    print("INICIANDO PIPELINE DE CONTAGEM AEREA DE BOVINOS")
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

    # Janela de exibição ao vivo (redimensionável, proporção do vídeo preservada)
    if show_window:
        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(
            config.WINDOW_NAME,
            int(config.WINDOW_HEIGHT * width / height),
            config.WINDOW_HEIGHT,
        )
        print("Janela ao vivo:   [ESPACO] pausa | [F] tela cheia | [Q]/[ESC] encerra")
    fullscreen = False

    # Contagens iniciais (garante o resumo final mesmo se nenhum frame for lido)
    counts: Dict[str, int] = {
        "active_in_frame": 0,
        "confirmed_unique_total": 0,
        "line_in": 0,
        "line_out": 0,
    }

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

                # Exibição ao vivo com controles de teclado
                if show_window:
                    cv2.imshow(config.WINDOW_NAME, annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(" "):
                        # Pausa até nova tecla de espaço (ou saída)
                        key = 0
                        while key not in (ord(" "), ord("q"), 27):
                            key = cv2.waitKey(50) & 0xFF
                    if key in (ord("f"), ord("F")):
                        fullscreen = not fullscreen
                        cv2.setWindowProperty(
                            config.WINDOW_NAME,
                            cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                        )
                    window_closed = cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
                    if key in (ord("q"), 27) or window_closed:
                        print("\n[INFO] Exibição encerrada pelo usuário.")
                        break
    finally:
        # Liberação garantida de recursos para evitar corrupção de arquivo de vídeo
        cap.release()
        writer.release()
        if show_window:
            cv2.destroyAllWindows()
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
    parser.add_argument(
        "--show",
        action="store_true",
        default=config.SHOW_WINDOW,
        help="Exibe o vídeo anotado ao vivo numa janela enquanto processa.",
    )

    args = parser.parse_args()

    process_video(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model_path=args.model,
        enable_linezone=args.enable_linezone,
        show_window=args.show,
    )
