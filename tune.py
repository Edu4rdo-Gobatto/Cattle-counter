"""
Benchmark de Parâmetros do Tracker (tune.py)
Reproduz o mesmo clipe com diferentes combinações de parâmetros do ByteTrack e
compara a qualidade da contagem — sem rodar o YOLO de novo (usa o cache de
detecções do validate.py).

Métricas por combinação:
  - confirmados            total de IDs que atingiram a persistência mínima
  - erro vs. referência     (confirmados - manual) / manual
  - IDs/confirmado          fragmentação (menor é melhor)
  - pico simultâneo         máximo de animais no quadro ao mesmo tempo
  - fragmentos recuperáveis IDs que viveram 4..14 frames (animal real perdido)

Uso:
    python tune.py --input video-gado.mp4 --limit 150 --gt 24
    python tune.py --input video-gado.mp4 --limit 150 --activations 0.30,0.25,0.20
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import config
from counter import CattleCounter
from tracker import CattleTracker
from validate import (
    VALIDATION_DIR,
    detect_frames,
    ground_truth_scope,
    load_ground_truth,
)


def parse_floats(text: str):
    return [float(v) for v in text.split(",") if v.strip()]


def parse_ints(text: str):
    return [int(v) for v in text.split(",") if v.strip()]


def evaluate(
    detections,
    fps: float,
    width: int,
    height: int,
    activation: float,
    matching: float,
    lost_buffer: int,
    min_persistence: int,
    tolerance: int = 0,
) -> dict:
    """Replay das detecções com uma combinação de parâmetros do tracker."""
    tracker = CattleTracker(
        track_activation_threshold=activation,
        lost_track_buffer=lost_buffer,
        minimum_matching_threshold=matching,
        frame_rate=int(fps),
    )
    counter = CattleCounter(
        frame_width=width,
        frame_height=height,
        min_persistence=min_persistence,
        streak_gap_tolerance=tolerance,
    )

    active_history = []
    for det in detections:
        tracked = tracker.update(det)
        counts = counter.update(tracked)
        active_history.append(counts["active_in_frame"])

    # Total final sem os objetos fixos na tela (marca d'água confirmada pela persistência)
    static_ids = counter.static_track_ids()
    confirmed = counter.unique_total()
    total_ids = len(counter.id_frame_counts)
    fragments = sum(
        1
        for frames in counter.id_frame_counts.values()
        if 4 <= frames < min_persistence
    )
    return {
        "activation": activation,
        "matching": matching,
        "tol": tolerance,
        "confirmed": confirmed,
        "raw": counts["confirmed_unique_total"],
        "static": len(static_ids),
        "total_ids": total_ids,
        "frag": (total_ids / confirmed) if confirmed else float("inf"),
        "peak": max(active_history) if active_history else 0,
        "fragments": fragments,
        "junk": sum(1 for v in counter.id_frame_counts.values() if v <= 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid de parâmetros do ByteTrack sobre detecções cacheadas."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument(
        "--end-frame", type=int, default=None,
        help="Último frame exclusivo (corta a vinheta, igual ao main.py). "
             "Tem prioridade sobre --limit.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.15,
        help="Confiança do cache de detecções (baixa de propósito: o tracker "
             "precisa ver as detecções fracas para reativar tracks perdidos).",
    )
    parser.add_argument("--iou", type=float, default=config.IOU_THRESHOLD)
    parser.add_argument("--imgsz", type=int, default=config.INFERENCE_IMAGE_SIZE)
    parser.add_argument("--activations", default="0.30,0.25,0.20")
    parser.add_argument("--matchings", default="0.70,0.60,0.50")
    parser.add_argument("--lost-buffer", type=int, default=config.LOST_TRACK_BUFFER)
    parser.add_argument("--min-persistence", type=int, default=config.MIN_FRAMES_PERSISTENCE)
    parser.add_argument("--tolerances", default=str(config.STREAK_GAP_TOLERANCE),
                        help="Tolerância(s) a frames ausentes na persistência.")
    parser.add_argument("--gt", type=int, default=None, help="Contagem manual de referência.")
    parser.add_argument("--no-gt-file", action="store_true",
                        help="Ignora validation/ground_truth.json.")
    args = parser.parse_args()

    input_path = Path(args.input)
    gt = args.gt
    gt_kind = "informada em --gt"
    if gt is None and not args.no_gt_file:
        gt_total = load_ground_truth(input_path.name, kind="total")
        gt_peak = load_ground_truth(input_path.name, kind="peak")
        if gt_total is not None:
            gt, gt_kind = gt_total, "total distinto"
        elif gt_peak is not None:
            gt, gt_kind = gt_peak, "pico por frame"
    if gt:
        print(f"[Referência] {gt} bovinos ({gt_kind})")

    limit = args.limit
    if args.end_frame is not None:
        limit = max(0, args.end_frame - args.start)

    cap = cv2.VideoCapture(str(input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    detections, _, fps = detect_frames(
        input_path,
        start=args.start,
        limit=limit,
        confidence=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )
    if not detections:
        raise SystemExit("Nenhum frame processado.")

    # O erro só é comparável se a execução cobrir a mesma faixa da contagem manual.
    end = args.start + len(detections)
    if gt and gt_kind == "total distinto":
        scope = ground_truth_scope(input_path.name)
        if scope is None:
            print(
                "[AVISO] A referência de total distinto não registra a faixa de frames "
                "coberta (`validate.py gt --end N`): confira se a execução é o vídeo inteiro."
            )
        elif (args.start, end) != scope:
            print(
                f"[AVISO] A referência cobre os frames {scope[0]}-{scope[1]}, mas esta "
                f"execução cobre {args.start}-{end}. O erro abaixo NÃO é comparável."
            )

    results = []
    for activation in parse_floats(args.activations):
        for matching in parse_floats(args.matchings):
            for tolerance in parse_ints(args.tolerances):
                results.append(
                    evaluate(
                        detections, fps, width, height,
                        activation, matching, args.lost_buffer,
                        args.min_persistence, tolerance,
                    )
                )

    # Tabela
    print(f"\n{'=' * 88}")
    print(f"GRID DO TRACKER — {input_path.name} [{args.start}:{args.start + len(detections)}]"
          f" | conf cache {args.conf:.2f} | persistência {args.min_persistence}")
    print(f"{'=' * 88}")
    header = (
        f"{'ativ':>5} {'match':>6} {'tol':>4} | {'conf':>5} {'est':>4} {'IDs':>5} {'frag':>5} "
        f"{'pico':>5} {'fragm':>6} {'lixo':>5} | {'erro':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if gt:
            erro = f"{(r['confirmed'] - gt) / gt * 100:+.0f}%"
        else:
            erro = "  -"
        print(
            f"{r['activation']:>5.2f} {r['matching']:>6.2f} {r['tol']:>4} | "
            f"{r['confirmed']:>5} {r['static']:>4} {r['total_ids']:>5} {r['frag']:>5.2f} "
            f"{r['peak']:>5} {r['fragments']:>6} {r['junk']:>5} | {erro:>7}"
        )

    # Ranking: menor erro absoluto vs. referência; empate -> menor fragmentação
    valid = [r for r in results if np.isfinite(r["frag"])]
    if gt and valid:
        best = min(valid, key=lambda r: (abs(r["confirmed"] - gt), r["frag"]))
        print(
            f"\nMelhor combinação vs. referência: ativ {best['activation']:.2f} / "
            f"match {best['matching']:.2f} -> {best['confirmed']} confirmados "
            f"(erro {(best['confirmed'] - gt) / gt * 100:+.0f}%, frag {best['frag']:.2f})"
        )
    elif valid:
        best = min(valid, key=lambda r: r["frag"])
        print(
            f"\nMenor fragmentação: ativ {best['activation']:.2f} / "
            f"match {best['matching']:.2f} -> frag {best['frag']:.2f} "
            f"({best['confirmed']} confirmados)"
        )
        print("Sem referência manual: rode 'python validate.py gt' para medir precisão.")

    VALIDATION_DIR.mkdir(exist_ok=True)
    csv_path = VALIDATION_DIR / (
        f"tune_{input_path.stem}_f{args.start}_{len(detections)}_c{args.conf:.2f}.csv"
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Resultados salvos em: {csv_path}")


if __name__ == "__main__":
    main()
