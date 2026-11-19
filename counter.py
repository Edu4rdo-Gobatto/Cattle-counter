"""
Módulo Counter (counter.py)
Implementa a lógica de contagem do rebanho:
1. Contagem Ativa no Frame (Animais visíveis agora - Imune a giros 360°)
2. Acúmulo de IDs Únicos com Filtro de Persistência (Evita ruídos e falsos positivos de vegetação/sombras)
3. LineZone opcional com detecção de sentido (In/Out) para voos retilíneos em corredores.
"""

from typing import Dict, List, Set, Tuple, Optional
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
        streak_gap_tolerance: int = config.STREAK_GAP_TOLERANCE,
        static_max_spread: float = config.STATIC_TRACK_MAX_SPREAD,
        static_min_frames: int = config.STATIC_TRACK_MIN_FRAMES,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.min_persistence = min_persistence
        self.enable_linezone = enable_linezone
        self.streak_gap_tolerance = streak_gap_tolerance
        self.static_max_spread = static_max_spread
        self.static_min_frames = static_min_frames

        # Dicionário para rastrear a sequência atual de frames de cada ID
        # (persistência: reseta quando o ID fica mais que N frames fora do quadro)
        self.id_frame_streaks: Dict[int, int] = {}

        # Quantos frames consecutivos cada ID passou ausente (tolerância a falhas)
        self.id_missed_frames: Dict[int, int] = {}

        # Dicionário para rastrear há quantos frames cada ID foi observado
        # (usado em diagnósticos de fragmentação no pipeline principal)
        self.id_frame_counts: Dict[int, int] = {}

        # Diagnóstico por ID: em quais frames ele apareceu e qual foi o maior
        # streak consecutivo alcançado (base do relatório do validate.py)
        self.id_appearance_frames: Dict[int, List[int]] = {}
        self.id_longest_streak: Dict[int, int] = {}
        self._frame_index = 0

        # Extremos do centro da caixa por ID: [min_x, max_x, min_y, max_y].
        # Base do filtro de objeto fixo na tela (marca d'água, logo, poste).
        self.id_center_bounds: Dict[int, List[float]] = {}

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
        self._frame_index += 1

        if tracked_detections.tracker_id is not None:
            for index, tracker_id in enumerate(tracked_detections.tracker_id):
                tracker_id = int(tracker_id)
                active_ids.add(tracker_id)
                self._update_center_bounds(tracker_id, tracked_detections.xyxy[index])

                # Total de frames em que o ID já foi visto (diagnóstico)
                self.id_frame_counts[tracker_id] = (
                    self.id_frame_counts.get(tracker_id, 0) + 1
                )
                self.id_appearance_frames.setdefault(tracker_id, []).append(
                    self._frame_index
                )

                # Persistência CONSECUTIVA: o streak zera quando o ID some do frame
                streak = self.id_frame_streaks.get(tracker_id, 0) + 1
                self.id_frame_streaks[tracker_id] = streak
                self.id_longest_streak[tracker_id] = max(
                    self.id_longest_streak.get(tracker_id, 0), streak
                )

                # Só confirma o animal após N frames consecutivos de rastreamento
                if streak >= self.min_persistence:
                    self.confirmed_unique_ids.add(tracker_id)

        # Ausências curtas (oclusão, falha de detecção) NÃO zeram o streak:
        # o ByteTrack costuma reativar o mesmo ID alguns frames depois. Só zera
        # quando a ausência passa da tolerância configurada.
        for tracker_id in self.id_frame_streaks:
            if tracker_id in active_ids:
                self.id_missed_frames[tracker_id] = 0
            else:
                missed = self.id_missed_frames.get(tracker_id, 0) + 1
                self.id_missed_frames[tracker_id] = missed
                if missed > self.streak_gap_tolerance:
                    self.id_frame_streaks[tracker_id] = 0

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

    def _update_center_bounds(self, tracker_id: int, box: np.ndarray) -> None:
        """Guarda os extremos do centro da caixa para detectar objeto fixo na tela."""
        center_x = float(box[0] + box[2]) / 2
        center_y = float(box[1] + box[3]) / 2
        bounds = self.id_center_bounds.get(tracker_id)
        if bounds is None:
            self.id_center_bounds[tracker_id] = [center_x, center_x, center_y, center_y]
            return
        bounds[0] = min(bounds[0], center_x)
        bounds[1] = max(bounds[1], center_x)
        bounds[2] = min(bounds[2], center_y)
        bounds[3] = max(bounds[3], center_y)

    def static_track_ids(self) -> Set[int]:
        """
        IDs confirmados que na verdade são objetos fixos na tela (marca d'água).

        Só descarta quando o centro variou quase nada E o ID viveu muito tempo:
        um bovino se desloca com a câmera, uma sobreposição de tela não.
        """
        static_ids: Set[int] = set()
        for tracker_id in self.confirmed_unique_ids:
            bounds = self.id_center_bounds.get(tracker_id)
            if bounds is None:
                continue
            spread = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
            if (
                spread <= self.static_max_spread
                and self.id_frame_counts.get(tracker_id, 0) >= self.static_min_frames
            ):
                static_ids.add(tracker_id)
        return static_ids

    def unique_total(self) -> int:
        """Total final: confirmados menos os objetos fixos na tela."""
        return len(self.confirmed_unique_ids - self.static_track_ids())

    def annotate_linezone(self, frame: np.ndarray) -> np.ndarray:
        """Desenha a LineZone no frame caso esteja habilitada."""
        if self.enable_linezone and self.line_zone and self.line_zone_annotator:
            return self.line_zone_annotator.annotate(frame, line_counter=self.line_zone)
        return frame
