"""
Pseudo-rotulagem automática (pseudo_label.py)
Etapa 0 do docs/PLANO-TREINAMENTO.md, opção B (T2).

Gera rótulos YOLO a partir do próprio `yolo11ft` (conf >= PSEUDO_CONF),
sem nenhuma caixa desenhada à mão (regra 1). Serve SÓ para o treino;
a avaliação usa dataset público anotado por terceiros + contagem humana.

Separação por vídeo: cada vídeo vira uma subpasta (images/<video>/ e
labels/<video>/), para não vazar frames quase idênticos entre treino e val.

Uso:
    python pseudo_label.py --input gado.webm --output data/pseudo
    python pseudo_label.py --input video.mp4 --output data/pseudo --every 15 --conf 0.5
"""

import argparse
from pathlib import Path

import cv2

import config
from detector import CattleDetector


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera pseudo-rótulos YOLO a partir do próprio detector."
    )
    parser.add_argument("--input", type=str, required=True, help="Vídeo de entrada.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/pseudo",
        help="Pasta base (cria images/<video>/ e labels/<video>/).",
    )
    parser.add_argument("--every", type=int, default=config.PSEUDO_EVERY_N)
    parser.add_argument("--conf", type=float, default=config.PSEUDO_CONF)
    parser.add_argument("--iou", type=float, default=config.IOU_THRESHOLD)
    parser.add_argument("--imgsz", type=int, default=config.INFERENCE_IMAGE_SIZE)
    parser.add_argument("--model", type=str, default=config.MODEL_WEIGHTS)
    return parser.parse_args()


def to_yolo_line(box, img_w: int, img_h: int, cls: int = 0) -> str:
    """Converte uma caixa xyxy em linha YOLO normalizada (cls cx cy w h)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main(
    input_path: Path,
    output_base: Path,
    every: int = config.PSEUDO_EVERY_N,
    conf: float = config.PSEUDO_CONF,
    iou: float = config.IOU_THRESHOLD,
    imgsz: int = config.INFERENCE_IMAGE_SIZE,
    model: str = config.MODEL_WEIGHTS,
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {input_path}")

    detector = CattleDetector(
        model_path=model, confidence=conf, iou=iou, imgsz=imgsz
    )
    tag = input_path.stem
    img_dir = output_base / "images" / tag
    lbl_dir = output_base / "labels" / tag
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise IOError(f"Falha ao abrir o vídeo: {input_path}")

    saved = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every == 0:
            det = detector.detect(frame)
            h, w = frame.shape[:2]
            lines = []
            for i, box in enumerate(det.xyxy):
                cls = int(det.class_id[i]) if det.class_id is not None else 0
                lines.append(to_yolo_line(box, w, h, cls))
            name = f"{tag}_f{idx:05d}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), frame)
            (lbl_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
            saved += 1
        idx += 1
    cap.release()

    print(f"[Pseudo] {saved} frames salvos de {input_path.name} (1 a cada {every}).")
    print(f"[Pseudo] imagens: {img_dir} | rótulos: {lbl_dir}")
    print("[Pseudo] Lembrete: usar SÓ no treino (A+B), nunca para avaliar.")
    return saved


if __name__ == "__main__":
    args = build_args()
    main(
        input_path=Path(args.input),
        output_base=Path(args.output),
        every=args.every,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        model=args.model,
    )
