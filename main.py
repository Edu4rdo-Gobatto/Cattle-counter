"""
Pipeline Principal (main.py)
Orquestra o fluxo de processamento de vídeo offline:
Decodificação -> Detecção YOLO -> Rastreamento ByteTrack -> Contagem -> HUD & Anotação -> Gravação
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
    use_slicing: bool = config.USE_SLICING,
    confidence: float = config.CONFIDENCE_THRESHOLD,
    iou: float = config.IOU_THRESHOLD,
    imgsz: int = config.INFERENCE_IMAGE_SIZE,
    track_activation: float = config.TRACK_ACTIVATION_THRESHOLD,
    min_matching: float = config.MINIMUM_MATCHING_THRESHOLD,
    streak_tolerance: int = config.STREAK_GAP_TOLERANCE,
    end_frame: Optional[int] = config.END_FRAME,
):
    """Executa o pipeline completo de contagem aérea e gravação do vídeo anotado."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"\n[ERRO CRÍTICO] Vídeo de entrada não encontrado em:\n{input_path}\n"
            f"Por favor, coloque o vídeo no diretório do projeto ou especifique via argumento --input."
        )

    model_file = Path(model_path)
    if model_file.parent != Path(".") and not model_file.exists():
        raise FileNotFoundError(
            f"\n[ERRO] Modelo não encontrado em: {model_file}\n"
            "Baixe o modelo especializado e salve em 'models/yolo11ft.pt':\n"
            "  https://github.com/xxilytoo/cattle_graze_model\n"
            "Ou rode com o modelo genérico: --model yolov8n.pt "
            "(nesse caso ajuste CLASSES_OF_INTEREST para [18, 19] no config.py)."
        )

    print(f"\n{'='*70}")
    print("INICIANDO PIPELINE DE CONTAGEM AEREA DE BOVINOS")
    print(f"{'='*70}")
    print(f"Vídeo de Entrada: {input_path}")
    print(f"Vídeo de Saída:   {output_path}")
    print(f"Modelo:           {model_path}")
    print(f"Confiança / IoU:  {confidence:.2f} / {iou:.2f} | imgsz {imgsz}")
    print(f"Tracker:          ativação {track_activation:.2f} | matching {min_matching:.2f} | "
          f"tolerância de falha {streak_tolerance}f")
    print(f"Modo LineZone:    {'ATIVADO' if enable_linezone else 'DESATIVADO (Rastreamento por Persistência)'}")
    print(f"Modo Fatiamento:  {'ATIVADO (SAHI)' if use_slicing else 'DESATIVADO'}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise IOError(f"Falha ao abrir o arquivo de vídeo: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Corte opcional do fim do vídeo (vinhetas, créditos)
    if end_frame is not None and 0 < end_frame < total_frames:
        total_frames = end_frame

    print(f"Resolução:        {width}x{height} @ {fps:.2f} FPS")
    print(f"Total de Frames:  {total_frames}" + (f" (corte em --end-frame {end_frame})" if end_frame else ""))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    detector = CattleDetector(
        model_path=model_path,
        use_slicing=use_slicing,
        confidence=confidence,
        iou=iou,
        imgsz=imgsz,
    )
    tracker = CattleTracker(
        frame_rate=int(fps),
        track_activation_threshold=track_activation,
        minimum_matching_threshold=min_matching,
    )
    counter = CattleCounter(
        frame_width=width,
        frame_height=height,
        enable_linezone=enable_linezone,
        streak_gap_tolerance=streak_tolerance,
    )

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

    frame_idx = 0
    start_total_time = time.time()
    last_time = start_total_time
    proc_fps = 0.0

    try:
        with tqdm(total=total_frames, desc="Processando Vídeo", unit="frame") as pbar:
            while end_frame is None or frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                t_now = time.time()
                elapsed_frame = t_now - last_time
                last_time = t_now
                if elapsed_frame > 0:
                    proc_fps = 1.0 / elapsed_frame

                detections = detector.detect(frame)

                tracked_detections = tracker.update(detections)

                counts = counter.update(tracked_detections)

                annotated_frame = frame.copy()

                annotated_frame = trace_annotator.annotate(
                    scene=annotated_frame,
                    detections=tracked_detections,
                )

                annotated_frame = box_annotator.annotate(
                    scene=annotated_frame,
                    detections=tracked_detections,
                )

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

                annotated_frame = counter.annotate_linezone(annotated_frame)

                annotated_frame = draw_hud(
                    frame=annotated_frame,
                    counts=counts,
                    current_frame=frame_idx,
                    total_frames=total_frames,
                    current_fps=proc_fps,
                    enable_linezone=enable_linezone,
                )

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

    # Objetos fixos na tela (marca d'água/logo) confirmam pela persistência, mas
    # não são animais: saem do total final. Só dá para decidir no fim do vídeo.
    static_ids = counter.static_track_ids()
    confirmed_total = counts["confirmed_unique_total"]
    final_total = counter.unique_total()
    print(f"Bovinos Únicos Validados: {final_total}")
    if static_ids:
        print(f"Descartados (fixos na tela): {len(static_ids)} "
              f"-> IDs {sorted(static_ids)} (confirmados brutos: {confirmed_total})")

    # Diagnóstico de fragmentação: muitos IDs criados para poucos confirmados indica
    # que o tracker está perdendo animais e recriando IDs novos (blur, oclusão, câmera)
    total_ids = len(counter.id_frame_counts)
    print(f"IDs Criados pelo Tracker: {total_ids}")
    if final_total > 0:
        print(f"Fragmentação (IDs/conf.): {total_ids / final_total:.2f}")
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
    parser.add_argument(
        "--slicing",
        action="store_true",
        default=config.USE_SLICING,
        help="Ativa o fatiamento SAHI (mais detecção de animais pequenos, ~8x mais lento).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help="Confiança mínima da detecção (menor = pega animais distantes, mas gera ruído).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=config.IOU_THRESHOLD,
        help="IoU do NMS (maior = preserva caixas sobrepostas de animais agrupados).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=config.INFERENCE_IMAGE_SIZE,
        help="Resolução de inferência (maior = detecta animais pequenos/distantes, mais lento).",
    )
    parser.add_argument(
        "--track-act",
        type=float,
        default=config.TRACK_ACTIVATION_THRESHOLD,
        help="Confiança mínima para ativar um track no ByteTrack.",
    )
    parser.add_argument(
        "--min-match",
        type=float,
        default=config.MINIMUM_MATCHING_THRESHOLD,
        help="Limiar de associação do ByteTrack (maior = aceita IoU menor, menos fragmentação).",
    )
    parser.add_argument(
        "--streak-tol",
        type=int,
        default=config.STREAK_GAP_TOLERANCE,
        help="Frames ausentes tolerados antes de zerar a persistência de um ID.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=config.END_FRAME,
        help="Para no frame N (exclusivo), ex.: 491 corta a vinheta do video-gado.mp4.",
    )

    args = parser.parse_args()

    process_video(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model_path=args.model,
        enable_linezone=args.enable_linezone,
        show_window=args.show,
        use_slicing=args.slicing,
        confidence=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        track_activation=args.track_act,
        min_matching=args.min_match,
        streak_tolerance=args.streak_tol,
        end_frame=args.end_frame,
    )
