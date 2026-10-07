"""
Módulo Tracker (tracker.py)
Encapsula o algoritmo ByteTrack utilizando a biblioteca supervision.
Gerencia a persistência espacial de IDs através de predições de Filtro de Kalman.
"""

import supervision as sv
import config


class CattleTracker:
    """Gerenciador de rastreamento de instâncias com ByteTrack."""

    def __init__(
        self,
        track_activation_threshold: float = config.TRACK_ACTIVATION_THRESHOLD,
        lost_track_buffer: int = config.LOST_TRACK_BUFFER,
        minimum_matching_threshold: float = config.MINIMUM_MATCHING_THRESHOLD,
        frame_rate: int = config.DEFAULT_FPS,
    ):
        print(
            f"[Tracker] Inicializando ByteTrack (LostBuffer: {lost_track_buffer}, "
            f"ActivationThresh: {track_activation_threshold})"
        )
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Atualiza o estado do rastreador com as novas detecções do quadro atual.
        
        Args:
            detections: Objeto supervision.Detections do frame atual.
            
        Returns:
            supervision.Detections: Detecções contendo o atributo 'tracker_id' preenchido.
        """
        tracked_detections = self.tracker.update_with_detections(detections)
        return tracked_detections

    def reset(self):
        """Reinicia os estados internos do tracker."""
        self.tracker.reset()
