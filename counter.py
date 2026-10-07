"""
Módulo Counter (counter.py)
Implementa a lógica de contagem do rebanho:
1. Contagem Ativa no Frame (Animais visíveis agora - Imune a giros 360°)
2. Acúmulo de IDs Únicos com Filtro de Persistência (Evita ruídos e falsos positivos de vegetação/sombras)
3. LineZone opcional com detecção de sentido (In/Out) para voos retilíneos em corredores.
"""

from typing import Dict, Set, Tuple, Optional
import numpy as np
import supervision as sv

import config


class CattleCounter:
    """Gerenciador de contagem inteligente de rebanho."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        min_persistence: int = config.MIN_FRAMES_PERSISTENCE,
        enable_linezone: bool = config.ENABLE_LINEZONE,
        line_start_rel: Tuple[float, float] = config.LINEZONE_START_RELATIVE,
        line_end_rel: Tuple[float, float] = config.LINEZONE_END_RELATIVE,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.min_persistence = min_persistence
        self.enable_linezone = enable_linezone

        # Dicionário para rastrear há quantos frames cada ID foi observado
        self.id_frame_counts: Dict[int, int] = {}

        # Conjunto de IDs confirmados que atingiram o limiar de persistência
        self.confirmed_unique_ids: Set[int] = set()

        # Configuração opcional da LineZone
        self.line_zone: Optional[sv.LineZone] = None
        self.line_zone_annotator: Optional[sv.LineZoneAnnotator] = None

        if self.enable_linezone:
            start_point = sv.Point(
                x=int(frame_width * line_start_rel[0]),
                y=int(frame_height * line_start_rel[1]),
            )
            end_point = sv.Point(
                x=int(frame_width * line_end_rel[0]),
                y=int(frame_height * line_end_rel[1]),
            )
            self.line_zone = sv.LineZone(start=start_point, end=end_point)
            self.line_zone_annotator = sv.LineZoneAnnotator(
                thickness=2,
                color=sv.Color.from_hex("#00FFAA"),
                text_scale=0.7,
                text_thickness=2,
            )
            print(f"[Counter] LineZone configurada de {start_point} até {end_point}")

    def update(self, tracked_detections: sv.Detections) -> Dict[str, int]:
        """
        Processa as detecções rastreadas do frame atual e atualiza contadores.
        
        Args:
            tracked_detections: supervision.Detections contendo 'tracker_id'.
            
        Returns:
            Dict com:
                'active_in_frame': quantidade de animais no frame atual
                'confirmed_unique_total': total de IDs únicos confirmados
                'line_in': contagem de entrada da LineZone (se ativada)
                'line_out': contagem de saída da LineZone (se ativada)
        """
        active_ids = set()

        if tracked_detections.tracker_id is not None:
            for tracker_id in tracked_detections.tracker_id:
                active_ids.add(int(tracker_id))
                self.id_frame_counts[tracker_id] = (
                    self.id_frame_counts.get(tracker_id, 0) + 1
                )

                # Se o animal persistiu pelo tempo mínimo necessário, é computado
                if self.id_frame_counts[tracker_id] >= self.min_persistence:
                    self.confirmed_unique_ids.add(int(tracker_id))

        # Atualiza a LineZone caso esteja habilitada
        line_in = 0
        line_out = 0
        if self.enable_linezone and self.line_zone is not None:
            self.line_zone.trigger(tracked_detections)
            line_in = self.line_zone.in_count
            line_out = self.line_zone.out_count

        return {
            "active_in_frame": len(active_ids),
            "confirmed_unique_total": len(self.confirmed_unique_ids),
            "line_in": line_in,
            "line_out": line_out,
        }

    def annotate_linezone(self, frame: np.ndarray) -> np.ndarray:
        """Desenha a LineZone no frame caso esteja habilitada."""
        if self.enable_linezone and self.line_zone and self.line_zone_annotator:
            return self.line_zone_annotator.annotate(frame, line_counter=self.line_zone)
        return frame
