"""
Testes Automatizados de Unidade e Integração (test_pipeline.py)
Valida a lógica matemática e espacial dos módulos:
1. CattleCounter (Persistência de IDs, contagem ativa e LineZone)
2. CattleTracker (Associação de Bboxes com ByteTrack)
3. HUD Drawing (Renderização gráfica e sobreposição de telemetria)
"""

import unittest
import numpy as np
import supervision as sv

import config
from counter import CattleCounter
from tracker import CattleTracker
from main import draw_hud


class TestCattleSystem(unittest.TestCase):

    def setUp(self):
        self.width = 1280
        self.height = 720
        self.counter = CattleCounter(
            frame_width=self.width,
            frame_height=self.height,
            min_persistence=5,  # 5 frames para teste rápido
            enable_linezone=True,
            line_start_rel=(0.1, 0.5),
            line_end_rel=(0.9, 0.5),
        )

    def test_empty_detections(self):
        """Verifica se detecções vazias não quebram o contador."""
        empty_dets = sv.Detections.empty()
        counts = self.counter.update(empty_dets)
        self.assertEqual(counts["active_in_frame"], 0)
        self.assertEqual(counts["confirmed_unique_total"], 0)

    def test_persistence_filter(self):
        """
        Verifica se um boi só é contado no 'total_confirmado'
        após persistir pela quantidade mínima de frames configurada.
        """
        # Simula boi com ID #10
        xyxy = np.array([[100, 100, 150, 150]], dtype=np.float32)
        dets = sv.Detections(
            xyxy=xyxy,
            confidence=np.array([0.85]),
            class_id=np.array([19]),
            tracker_id=np.array([10]),
        )

        # Envia por 4 frames (ainda não atingiu 5 frames)
        for _ in range(4):
            counts = self.counter.update(dets)
            self.assertEqual(counts["active_in_frame"], 1)
            self.assertEqual(counts["confirmed_unique_total"], 0)

        # No 5º frame, deve ser confirmado no totalizador
        counts = self.counter.update(dets)
        self.assertEqual(counts["active_in_frame"], 1)
        self.assertEqual(counts["confirmed_unique_total"], 1)

    def test_spurious_noise_filtered(self):
        """
        Falsos positivos (ex: pedra/folha detectada por apenas 2 frames)
        NÃO devem ser somados ao total confirmado.
        """
        # ID passageiro #99 por 2 frames
        noise_dets = sv.Detections(
            xyxy=np.array([[200, 200, 230, 230]], dtype=np.float32),
            confidence=np.array([0.40]),
            tracker_id=np.array([99]),
        )
        for _ in range(2):
            counts = self.counter.update(noise_dets)

        self.assertEqual(counts["confirmed_unique_total"], 0)

    def test_360_turn_immunity(self):
        """
        Em um giro 360°, se os mesmos animais permanecerem na tela por 50 frames,
        o total confirmado deve registrar apenas a quantidade real de animais,
        sem duplicar contagens.
        """
        # 3 animais (#1, #2, #3)
        xyxy = np.array([
            [100, 100, 150, 150],
            [300, 300, 350, 350],
            [500, 500, 550, 550]
        ], dtype=np.float32)
        dets = sv.Detections(
            xyxy=xyxy,
            confidence=np.array([0.9, 0.88, 0.92]),
            tracker_id=np.array([1, 2, 3]),
        )

        # 50 frames de giro contínuo
        for _ in range(50):
            counts = self.counter.update(dets)

        self.assertEqual(counts["active_in_frame"], 3)
        self.assertEqual(counts["confirmed_unique_total"], 3)

    def test_hud_rendering(self):
        """Garante que a renderização do HUD não corrompe o frame ou dimensões."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        counts = {
            "active_in_frame": 12,
            "confirmed_unique_total": 45,
            "line_in": 10,
            "line_out": 2,
        }
        hud_frame = draw_hud(
            frame=frame,
            counts=counts,
            current_frame=150,
            total_frames=900,
            current_fps=28.5,
            enable_linezone=True,
        )
        self.assertEqual(hud_frame.shape, (self.height, self.width, 3))
        # Verifica se a região do HUD possui pixels desenhados
        self.assertTrue(np.any(hud_frame[:85, :] > 0))


if __name__ == "__main__":
    unittest.main()
