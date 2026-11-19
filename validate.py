"""
Módulo de Validação (validate.py)
Ferramentas de medição do pipeline, separadas da execução normal:

1. `frames`  -> extrai frames-chave anotados para contagem manual (referência)
2. `report`  -> relatório de fragmentação por ID (CSV + resumo no terminal)
3. `gt`      -> registra a contagem manual de referência em validation/ground_truth.json

As detecções podem ser cacheadas em cache/*.npz para que os experimentos de
parâmetros (tune.py) não precisem rodar o YOLO de novo a cada combinação.

Uso:
    python validate.py frames --input video-gado.mp4 --limit 150 --pick-best 3
    python validate.py report --input video-gado.mp4 --limit 150
    python validate.py gt --video video-gado.mp4 --frame 50 --count 24
    python validate.py gt --video video-gado.mp4 --total 70 --end 491 --note "contado assistindo"
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import supervision as sv

import config
from detector import CattleDetector
from tracker import CattleTracker
from counter import CattleCounter

VALIDATION_DIR = config.BASE_DIR / "validation"
CACHE_DIR = config.BASE_DIR / "cache"
GROUND_TRUTH_PATH = VALIDATION_DIR / "ground_truth.json"

# Chaves reservadas no ground_truth.json (não são números de frame)
TOTAL_DISTINCT_KEY = "total_distinct"
TOTAL_DISTINCT_FRAMES_KEY = "total_distinct_frames"
NOTE_KEY = "note"


# ==============================================================================
# CACHE DE DETECÇÕES
# ==============================================================================
def cache_path(
    input_path: Path,
    start: int,
    limit: int,
    confidence: float,
    iou: float,
    imgsz: int,
) -> Path:
    """Nome do arquivo de cache derivado de todos os parâmetros que mudam a detecção."""
    return CACHE_DIR / (
        f"{input_path.stem}_f{start}_{limit}"
        f"_c{confidence:.2f}_i{iou:.2f}_s{imgsz}.npz"
    )


def detect_frames(
    input_path: Path,
    start: int = 0,
    limit: int = 150,
    confidence: float = config.CONFIDENCE_THRESHOLD,
    iou: float = config.IOU_THRESHOLD,
    imgsz: int = config.INFERENCE_IMAGE_SIZE,
    use_cache: bool = True,
) -> Tuple[List[sv.Detections], int, float]:
    """
    Roda o detector em [start, start+limit) e devolve uma lista de Detections
    (uma por frame lido), além de fps e total de frames do vídeo.

    O resultado é cacheado em cache/*.npz: o YOLO roda uma vez e os
    experimentos de rastreamento podem ser repetidos à vontade.
    """
    path = cache_path(input_path, start, limit, confidence, iou, imgsz)

    if use_cache and path.exists():
        data = np.load(path, allow_pickle=False)
        xyxy = data["xyxy"]
        conf = data["conf"]
        cls = data["class_id"]
        frame_idx = data["frame_idx"]
        n_frames = int(data["n_frames"])
        fps = float(data["fps"])

        detections = []
        for i in range(n_frames):
            sel = frame_idx == i
            detections.append(
                sv.Detections(
                    xyxy=xyxy[sel],
                    confidence=conf[sel],
                    class_id=cls[sel],
                )
            )
        print(f"[Cache] Detecções carregadas de {path.name} ({n_frames} frames)")
        return detections, int(data["total_frames"]), fps

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise IOError(f"Falha ao abrir o vídeo: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    detector = CattleDetector(
        model_path=config.MODEL_WEIGHTS,
        confidence=confidence,
        iou=iou,
        imgsz=imgsz,
    )

    detections: List[sv.Detections] = []
    boxes: List[np.ndarray] = []
    confs: List[np.ndarray] = []
    classes: List[np.ndarray] = []
    frame_ids: List[np.ndarray] = []

    t0 = time.time()
    while len(detections) < limit:
        ret, frame = cap.read()
        if not ret:
            break
        det = detector.detect(frame)
        idx = len(detections)
        detections.append(det)
        boxes.append(det.xyxy)
        confs.append(
            det.confidence if det.confidence is not None
            else np.zeros(len(det), dtype=np.float32)
        )
        classes.append(
            det.class_id if det.class_id is not None
            else np.zeros(len(det), dtype=int)
        )
        frame_ids.append(np.full(len(det), idx, dtype=int))
    cap.release()

    elapsed = time.time() - t0
    print(
        f"[Detector] {len(detections)} frames em {elapsed:.1f}s "
        f"({len(detections) / elapsed:.2f} FPS)"
        if elapsed > 0 else "[Detector] 0 frames"
    )

    if use_cache and detections:
        CACHE_DIR.mkdir(exist_ok=True)
        np.savez_compressed(
            path,
            xyxy=np.concatenate(boxes) if boxes else np.zeros((0, 4)),
            conf=np.concatenate(confs) if confs else np.zeros((0,)),
            class_id=np.concatenate(classes) if classes else np.zeros((0,), dtype=int),
            frame_idx=np.concatenate(frame_ids) if frame_ids else np.zeros((0,), dtype=int),
            n_frames=np.array(len(detections)),
            total_frames=np.array(total_frames),
            fps=np.array(fps),
        )
        print(f"[Cache] Detecções salvas em {path.name}")

    return detections, total_frames, fps


# ==============================================================================
# RASTREAMENTO (replay das detecções)
# ==============================================================================
def track_all(
    detections: Sequence[sv.Detections],
    fps: float,
    trk: Optional[CattleTracker] = None,
    counter: Optional[CattleCounter] = None,
    frame_width: int = 0,
    frame_height: int = 0,
) -> Tuple[CattleTracker, CattleCounter, Dict[str, int], List[int]]:
    """Repassa as detecções pelo tracker/contador e devolve o estado final."""
    trk = trk or CattleTracker(frame_rate=int(fps))
    counter = counter or CattleCounter(
        frame_width=frame_width, frame_height=frame_height
    )

    counts: Dict[str, int] = {
        "active_in_frame": 0,
        "confirmed_unique_total": 0,
        "line_in": 0,
        "line_out": 0,
    }
    active_history: List[int] = []
    for det in detections:
        tracked = trk.update(det)
        counts = counter.update(tracked)
        active_history.append(counts["active_in_frame"])

    return trk, counter, counts, active_history


def id_stats(counter: CattleCounter) -> List[Dict[str, int]]:
    """
    Consolida, por ID, a sequência de frames em que ele apareceu.

    id_frame_counts guarda o total de frames vistos; aqui reconstruímos
    first/last/max_gap/longest_streak a partir dos streaks consecutivos.
    """
    rows = []
    static_ids = counter.static_track_ids()
    for tracker_id, frames_seen in sorted(counter.id_frame_counts.items()):
        appearances = counter.id_appearance_frames.get(tracker_id, [])
        gaps = [
            b - a - 1 for a, b in zip(appearances, appearances[1:])
        ]
        rows.append(
            {
                "tracker_id": tracker_id,
                "first_frame": appearances[0] if appearances else -1,
                "last_frame": appearances[-1] if appearances else -1,
                "frames_seen": frames_seen,
                "max_gap": max(gaps) if gaps else 0,
                "longest_streak": counter.id_longest_streak.get(tracker_id, frames_seen),
                "confirmed": int(tracker_id in counter.confirmed_unique_ids),
                "static": int(tracker_id in static_ids),
            }
        )
    return rows


# ==============================================================================
# SUBCOMANDO: frames (amostras para contagem manual)
# ==============================================================================
def pick_best_frames(reference_counts: Sequence[int], n: int, min_gap: int) -> List[int]:
    """Escolhe os N frames com mais detecções, separados por pelo menos min_gap."""
    order = sorted(range(len(reference_counts)), key=lambda i: reference_counts[i], reverse=True)
    chosen: List[int] = []
    for idx in order:
        if len(chosen) >= n:
            break
        if all(abs(idx - c) >= min_gap for c in chosen):
            chosen.append(idx)
    return sorted(chosen)


def cmd_frames(args) -> None:
    input_path = Path(args.input)
    detections, _, _ = detect_frames(
        input_path,
        start=args.start,
        limit=args.limit,
        confidence=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )
    counts = [len(d) for d in detections]
    chosen = pick_best_frames(counts, args.pick_best, args.min_gap)

    VALIDATION_DIR.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(input_path))
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    print(f"\nFrames escolhidos (mais detecções):")
    for local_idx in chosen:
        frame_number = args.start + local_idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()
        if not ok:
            print(f"  frame {frame_number}: falha na leitura")
            continue

        det = detections[local_idx]
        annotated = frame.copy()
        for i, (x1, y1, x2, y2) in enumerate(det.xyxy.astype(int), start=1):
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, str(i), (x1 + 2, max(y1 - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
            )

        header = (
            f"frame {frame_number} | detectados: {len(det)} | "
            f"max no clipe: {max(counts)}"
        )
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(
            annotated, header, (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

        out = VALIDATION_DIR / f"{input_path.stem}_f{frame_number:05d}.png"
        cv2.imwrite(str(out), annotated)
        print(f"  frame {frame_number}: {len(det)} detecções -> {out}")

    cap.release()
    print(
        "\nConte os bovinos na imagem (ignorando as caixas) e registre com:\n"
        f'  python validate.py gt --video {input_path.name} --frame <frame> --count <N>'
    )


# ==============================================================================
# SUBCOMANDO: report (fragmentação por ID)
# ==============================================================================
def cmd_report(args) -> None:
    input_path = Path(args.input)
    detections, _, fps = detect_frames(
        input_path,
        start=args.start,
        limit=args.limit,
        confidence=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )
    cap = cv2.VideoCapture(str(input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    trk, counter, counts, active_history = track_all(
        detections, fps, frame_width=width, frame_height=height
    )
    rows = id_stats(counter)

    VALIDATION_DIR.mkdir(exist_ok=True)
    csv_path = VALIDATION_DIR / f"{input_path.stem}_ids.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [
            "tracker_id", "first_frame", "last_frame", "frames_seen",
            "max_gap", "longest_streak", "confirmed",
        ])
        writer.writeheader()
        writer.writerows(rows)

    confirmed = counter.unique_total()
    raw_confirmed = counts["confirmed_unique_total"]
    static_ids = counter.static_track_ids()
    total_ids = len(counter.id_frame_counts)
    short_ids = sum(1 for r in rows if r["frames_seen"] <= 3)
    fragments = sum(1 for r in rows if 4 <= r["frames_seen"] < counter.min_persistence)
    max_concurrent = max(active_history) if active_history else 0
    avg_concurrent = float(np.mean(active_history)) if active_history else 0.0

    print(f"\n{'=' * 62}")
    print(f"RELATÓRIO DE FRAGMENTAÇÃO — {input_path.name} "
          f"[{args.start}:{args.start + len(detections)}]")
    print(f"{'=' * 62}")
    print(f"Frames analisados:           {len(detections)}")
    print(f"Média de bovinos por frame:  {avg_concurrent:.1f}")
    print(f"Máximo simultâneo (pico):    {max_concurrent}")
    print(f"IDs criados pelo tracker:    {total_ids}")
    print(f"IDs confirmados (>= {counter.min_persistence} frames): {confirmed}")
    if static_ids:
        print(f"Descartados (fixos na tela): {len(static_ids)}  "
              f"IDs {sorted(static_ids)}  (brutos: {raw_confirmed})")
    print(f"IDs curtos (<= 3 frames):    {short_ids}  (ruído, filtrado)")
    print(f"IDs fragmento (4-{counter.min_persistence - 1} frames):    {fragments}  "
          f"(animal real que não confirmou)")
    if confirmed > 0:
        print(f"Fragmentação (IDs/conf.):    {total_ids / confirmed:.2f}")
        print(f"Confirmados / pico:          {confirmed / max_concurrent:.2f}x" if max_concurrent else "")
    print(f"CSV detalhado:               {csv_path}")

    gt_total = load_ground_truth(input_path.name, kind="total")
    gt_peak = load_ground_truth(input_path.name, kind="peak")
    if gt_total:
        print(f"\nReferência humana (total):   {gt_total} bovinos")
        print(f"Erro da contagem:            "
              f"{(confirmed - gt_total) / gt_total * 100:+.0f}%")
    elif gt_peak:
        print(f"\nReferência humana (pico):    {gt_peak} bovinos no quadro")
        print(f"Detector no pico:            {max_concurrent} caixas "
              f"({max_concurrent / gt_peak * 100:.0f}% do visível)")
    elif max_concurrent:
        print(
            f"\nLeitura: com a câmera quase parada, o total confirmado deveria ficar "
            f"próximo do pico ({max_concurrent}). Sem referência humana não dá para "
            f"afirmar se está certo."
        )


# ==============================================================================
# SUBCOMANDO: gt (contagem manual de referência)
# ==============================================================================
def load_ground_truth_raw() -> Dict[str, Dict[str, object]]:
    """Conteúdo bruto do ground_truth.json (vazio se o arquivo não existir)."""
    if not GROUND_TRUTH_PATH.exists():
        return {}
    return json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))


def cmd_gt(args) -> None:
    VALIDATION_DIR.mkdir(exist_ok=True)
    data = load_ground_truth_raw()
    video = data.setdefault(args.video, {})

    if args.total is not None:
        video[TOTAL_DISTINCT_KEY] = int(args.total)
        if args.end is not None:
            video[TOTAL_DISTINCT_FRAMES_KEY] = [0, int(args.end)]
        if args.note:
            video[NOTE_KEY] = args.note
        print(
            f"Referência registrada: {args.video} -> total distinto de "
            f"{args.total} bovinos"
            + (f" (frames 0-{args.end})" if args.end is not None else "")
        )
    else:
        if args.frame is None or args.count is None:
            raise SystemExit(
                "Informe --frame e --count (contagem de um frame) ou "
                "--total (total de animais distintos do vídeo)."
            )
        video[str(args.frame)] = int(args.count)
        if args.note:
            video[NOTE_KEY] = args.note
        print(f"Referência registrada: {args.video} frame {args.frame} -> {args.count} bovinos")

    GROUND_TRUTH_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Arquivo: {GROUND_TRUTH_PATH}")


# ==============================================================================
# CLI
# ==============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validação e diagnóstico do pipeline de contagem de bovinos."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--input", required=True, help="Vídeo de entrada.")
        p.add_argument("--start", type=int, default=0, help="Primeiro frame.")
        p.add_argument("--limit", type=int, default=150, help="Quantos frames processar.")
        p.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD)
        p.add_argument("--iou", type=float, default=config.IOU_THRESHOLD)
        p.add_argument("--imgsz", type=int, default=config.INFERENCE_IMAGE_SIZE)

    p_frames = sub.add_parser("frames", help="Extrai frames-chave anotados.")
    common(p_frames)
    p_frames.add_argument("--pick-best", type=int, default=3)
    p_frames.add_argument("--min-gap", type=int, default=45)
    p_frames.set_defaults(func=cmd_frames)

    p_report = sub.add_parser("report", help="Relatório de fragmentação por ID.")
    common(p_report)
    p_report.set_defaults(func=cmd_report)

    p_gt = sub.add_parser("gt", help="Registra contagem manual de referência.")
    p_gt.add_argument("--video", required=True)
    p_gt.add_argument("--frame", type=int, default=None, help="Frame contado.")
    p_gt.add_argument("--count", type=int, default=None, help="Animais naquele frame.")
    p_gt.add_argument(
        "--total", type=int, default=None,
        help="Total de animais distintos do vídeo (referência para o erro de contagem).",
    )
    p_gt.add_argument(
        "--end", type=int, default=None,
        help="Último frame coberto pela contagem de --total (ex.: 491, antes da vinheta).",
    )
    p_gt.add_argument("--note", default=None, help="Observação sobre a contagem.")
    p_gt.set_defaults(func=cmd_gt)

    return parser


def load_ground_truth(video_name: str, kind: str = "peak") -> Optional[int]:
    """
    Referência manual registrada para o vídeo.

    kind="peak"  -> maior contagem por frame (piso do rebanho, comparável a janelas)
    kind="total" -> total de animais distintos do vídeo (comparável ao vídeo inteiro)
    """
    video = load_ground_truth_raw().get(video_name) or {}

    if kind == "total":
        total = video.get(TOTAL_DISTINCT_KEY)
        return int(total) if isinstance(total, (int, float)) else None

    frames = [int(v) for k, v in video.items() if k.isdigit()]
    return max(frames) if frames else None


def ground_truth_scope(video_name: str) -> Optional[Tuple[int, int]]:
    """Faixa de frames coberta pela contagem de total distinto, se registrada."""
    video = load_ground_truth_raw().get(video_name) or {}
    scope = video.get(TOTAL_DISTINCT_FRAMES_KEY)
    if isinstance(scope, list) and len(scope) == 2:
        return int(scope[0]), int(scope[1])
    return None


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
