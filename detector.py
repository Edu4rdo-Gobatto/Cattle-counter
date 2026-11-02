"""
Módulo Detector (detector.py)
Encapsula o modelo YOLO (Ultralytics) com otimização multithread para CPU.
Converte as predições diretamente para o formato supervision.Detections.
"""

from typing import Optional, List
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv

import config


class CattleDetector:
    """Detector de bovinos baseado em arquiteturas YOLO (v8/v11)."""

    def __init__(
        self,
        model_path: str = config.MODEL_WEIGHTS,
        device: str = config.DEVICE,
        num_threads: int = config.CPU_NUM_THREADS,
        confidence: float = config.CONFIDENCE_THRESHOLD,
        iou: float = config.IOU_THRESHOLD,
        imgsz: int = config.INFERENCE_IMAGE_SIZE,
        classes: Optional[List[int]] = config.CLASSES_OF_INTEREST,
        use_slicing: bool = config.USE_SLICING,
    ):
        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.classes = classes
        self.use_slicing = use_slicing

        # Otimização multithread para CPU Ryzen 7 5700U
        if self.device == "cpu" and num_threads > 0:
            torch.set_num_threads(num_threads)

        print(f"[Detector] Carregando modelo YOLO a partir de: {model_path} (Device: {self.device})")
        self.model = YOLO(model_path)
        print("[Detector] Modelo carregado com sucesso.")

        # SAHI (InferenceSlicer): fatia o frame em janelas 640x640 com sobreposição
        # e funde as caixas com NMS. Ajuda os animais pequenos, mas é ~8x mais lento.
        self.slicer = None
        if self.use_slicing:
            self.slicer = sv.InferenceSlicer(
                callback=self._predict,
                slice_wh=config.SLICE_SIZE,
                overlap_wh=config.SLICE_OVERLAP,
                iou_threshold=self.iou,
            )
            print(
                f"[Detector] SAHI ativado (fatia {config.SLICE_SIZE}px, "
                f"overlap {config.SLICE_OVERLAP}px)"
            )

    def _predict(self, image: np.ndarray) -> sv.Detections:
        """Executa inferência em uma imagem (frame inteiro ou fatia)."""
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.classes,
            verbose=False,
        )[0]

        # Conversão nativa e vetorizada para supervision
        return sv.Detections.from_ultralytics(results)

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Executa inferência em um único quadro de vídeo.
        
        Args:
            frame: Imagem BGR (numpy.ndarray)
            
        Returns:
            supervision.Detections: Estrutura vetorizada contendo bounding boxes,
                                   confianças e classes filtradas.
        """
        if self.slicer is not None:
            return self.slicer(frame)
        return self._predict(frame)
