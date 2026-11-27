"""
Treino mínimo com fine-tuning (train.py)
Etapa 0 do docs/PLANO-TREINAMENTO.md (T2).

Wrapper fino do `ultralytics.YOLO(...).train()`: todos os parâmetros vêm do
`config.py` (regra 6). Não desenha nenhuma caixa à mão (regra 1): o treino usa
dataset público já anotado (opção A) + pseudo-rótulos gerados pelo próprio
modelo (opção B, via pseudo_label.py).

Uso:
    python train.py --data data/datasets/aerial-cows/data.yaml
    python train.py --data data/datasets/aerial-cows/data.yaml --name ab --pseudodata data/pseudo/data.yaml
    python train.py --smoke  # 1 época, imgsz 640, batch 2: só valida o ambiente
"""

import argparse
from pathlib import Path
from typing import Optional

import config


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuning do YOLO para gado aéreo (transfer learning)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.MODEL_WEIGHTS,
        help="Pesos base (default: config.MODEL_WEIGHTS).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=config.TRAIN_DATA_YAML,
        help="data.yaml do dataset público anotado (opção A).",
    )
    parser.add_argument(
        "--pseudodata",
        type=str,
        default=None,
        help="data.yaml dos pseudo-rótulos (opção B). Se dado, combina A+B.",
    )
    parser.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=config.TRAIN_IMGSZ)
    parser.add_argument("--batch", type=int, default=config.TRAIN_BATCH)
    parser.add_argument("--freeze", type=int, default=config.TRAIN_FREEZE)
    parser.add_argument("--patience", type=int, default=config.TRAIN_PATIENCE)
    parser.add_argument("--project", type=str, default=config.TRAIN_PROJECT)
    parser.add_argument(
        "--name",
        type=str,
        default=config.TRAIN_NAME_A,
        help="Nome do experimento (a = só público, ab = público + pseudo).",
    )
    parser.add_argument("--seed", type=int, default=config.TRAIN_SEED)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 1 época, batch 2, para validar o ambiente em CPU.",
    )
    return parser.parse_args()


def main(
    model: str = config.MODEL_WEIGHTS,
    data: str = config.TRAIN_DATA_YAML,
    pseudodata: Optional[str] = None,
    epochs: int = config.TRAIN_EPOCHS,
    imgsz: int = config.TRAIN_IMGSZ,
    batch: int = config.TRAIN_BATCH,
    freeze: int = config.TRAIN_FREEZE,
    patience: int = config.TRAIN_PATIENCE,
    project: str = config.TRAIN_PROJECT,
    name: str = config.TRAIN_NAME_A,
    seed: int = config.TRAIN_SEED,
    smoke: bool = False,
) -> Path:
    """Roda o fine-tuning e devolve o caminho do best.pt gerado."""
    from ultralytics import YOLO

    if smoke:
        epochs, batch = 1, 2

    data_yaml = data
    if pseudodata:
        # A combinação A+B é feita via um data.yaml que lista os dois conjuntos.
        # Mantido como argumento explícito para não esconder dados no código.
        data_yaml = pseudodata
        print(f"[Train] A+B: usando data.yaml combinado: {data_yaml}")
    else:
        print(f"[Train] Opção A: dataset público anotado: {data_yaml}")

    print(
        f"[Train] base={model} epochs={epochs} imgsz={imgsz} batch={batch} "
        f"freeze={freeze} patience={patience} seed={seed}"
    )
    yolo = YOLO(model)
    yolo.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        freeze=freeze,
        patience=patience,
        project=project,
        name=name,
        seed=seed,
    )
    best = Path(project) / name / "weights" / "best.pt"
    print(f"[Train] Concluído. Pesos em: {best}")
    print("[Train] Avalie com o mesmo protocolo (Etapa 4 do plano) antes de trocar MODEL_WEIGHTS.")
    return best


if __name__ == "__main__":
    args = build_args()
    main(
        model=args.model,
        data=args.data,
        pseudodata=args.pseudodata,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        patience=args.patience,
        project=args.project,
        name=args.name,
        seed=args.seed,
        smoke=args.smoke,
    )
