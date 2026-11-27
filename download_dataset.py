"""
Baixa o dataset público anotado sem precisar de ROBOFLOW_API_KEY (T2, opção A).

Fonte: HuggingFace `Francesco/aerial-cows` (espelho do RF100 `aerial-cows`,
CC BY 4.0, 1084 train / 299 valid / 340 test, 640x640).
Formato original: parquet com `image` + `objects {bbox COCO, category}`.
Saída: YOLOv8 (`images/` + `labels/*.txt` + `data.yaml`), pronta para
`train.py` e `yolo val`.

Uso:
    .venv/bin/pip install -q datasets pillow
    .venv/bin/python download_dataset.py
    .venv/bin/python download_dataset.py --no-test  # pula o split test (mais rápido)
"""

import argparse
from pathlib import Path

import config

# Mantido aqui (não no config) porque é detalhe de fonte externa, não parâmetro
# do pipeline. O destino final respeita config.TRAIN_DATA_YAML.
HF_DATASET = "Francesco/aerial-cows"
DEST_BASE = Path(config.TRAIN_DATA_YAML).parent  # data/datasets/aerial-cows
NAMES = ["cow"]


def coco_to_yolo(bbox, img_w: int, img_h: int) -> str:
    """COCO [x, y, w, h] -> YOLO '0 cx cy w h' normalizado (classe única)."""
    x, y, w, h = (float(v) for v in bbox)
    return f"0 {(x + w / 2) / img_w:.6f} {(y + h / 2) / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}"


def main(skip_test: bool = False) -> Path:
    from datasets import load_dataset

    print(f"[Dataset] Baixando {HF_DATASET} (288 MB, 1723 linhas)...")
    ds = load_dataset(HF_DATASET)
    print(f"[Dataset] Splits: {list(ds.keys())}")

    splits = {"train": "train", "validation": "valid", "valid": "valid", "test": "test"}
    if skip_test:
        splits.pop("test", None)

    for hf_split, out_split in splits.items():
        if hf_split not in ds:
            print(f"[Dataset] Split {hf_split} ausente, pulando.")
            continue
        img_dir = DEST_BASE / out_split / "images"
        lbl_dir = DEST_BASE / out_split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for i, row in enumerate(ds[hf_split]):
            img = row["image"]
            w, h = img.size
            objs = row.get("objects") or {}
            bboxes = objs.get("bbox") or []
            lines = [coco_to_yolo(b, w, h) for b in bboxes]
            name = f"{hf_split}_{i:05d}"
            img.save(img_dir / f"{name}.jpg")
            (lbl_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
            n += 1
        print(f"[Dataset] {hf_split} -> {n} imagens em {img_dir.parent}")

    yaml_path = DEST_BASE / "data.yaml"
    yaml_path.write_text(
        f"path: {DEST_BASE.resolve()}\n"
        f"train: train/images\nval: valid/images\ntest: test/images\n\n"
        f"nc: 1\nnames: {NAMES}\n",
        encoding="utf-8",
    )
    print(f"[Dataset] data.yaml em {yaml_path}")
    print("[Dataset] Licença: CC BY 4.0 (Roboflow 100 aerial-cows).")
    print("Crédito: aerial-cows from Roboflow 100 (Roboflow Universe), CC BY 4.0.")
    return yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baixa aerial-cows (HF) em formato YOLO.")
    parser.add_argument("--no-test", action="store_true", help="Pula o split test.")
    args = parser.parse_args()
    main(skip_test=args.no_test)
