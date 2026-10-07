"""
Script de Geração de Vídeo Sintético de Teste (generate_test_video.py)
Cria um vídeo simulando visão aérea de drone sobrevoando um pasto com bovinos em movimento.
Utilizado para validar a integridade do pipeline sem necessidade imediata de vídeo real.
"""

import cv2
import numpy as np
from pathlib import Path


def create_synthetic_drone_video(
    output_path: str = "test_drone_video.mp4",
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration_sec: int = 10,
):
    total_frames = fps * duration_sec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[Gerador] Gerando vídeo de teste aéreo ({width}x{height}, {fps} FPS, {duration_sec}s)...")

    # Simulação de 6 animais no pasto com posições e vetores de velocidade
    np.random.seed(42)
    num_animals = 6
    animals = []
    colors = [
        (30, 45, 60),    # Bovino preto/angus
        (220, 220, 230), # Nelore branco/cinza claro
        (50, 70, 110),   # Castanho escuro
        (200, 200, 210), # Nelore branco
        (35, 50, 65),    # Preto
        (60, 80, 120),   # Castanho
    ]

    for i in range(num_animals):
        x = np.random.uniform(width * 0.15, width * 0.85)
        y = np.random.uniform(height * 0.2, height * 0.8)
        vx = np.random.uniform(-1.2, 1.2)
        vy = np.random.uniform(-0.8, 0.8)
        body_len = np.random.randint(28, 40)
        body_width = int(body_len * 0.5)
        animals.append({
            "x": x, "y": y, "vx": vx, "vy": vy,
            "len": body_len, "width": body_width,
            "color": colors[i % len(colors)]
        })

    # Textura base do pasto (verde com ruído de grama)
    base_grass = np.zeros((height, width, 3), dtype=np.uint8)
    base_grass[:, :] = (35, 95, 45) # BGR verde pastagem

    # Adiciona textura de terreno
    noise = np.random.normal(0, 12, (height, width, 3)).astype(np.int16)
    textured_grass = np.clip(base_grass.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Simulação de drone em translação e leve oscilação
    for frame_idx in range(total_frames):
        # Deslocamento sutil do drone (câmera avançando e oscilando pelo vento)
        cam_shift_x = int(np.sin(frame_idx * 0.03) * 6)
        cam_shift_y = int(frame_idx * 0.6) % 50

        frame = np.roll(textured_grass, cam_shift_y, axis=0)
        frame = np.roll(frame, cam_shift_x, axis=1)
        frame = frame.copy()

        # Atualiza e desenha cada bovino com sombra projetada
        for a in animals:
            a["x"] += a["vx"]
            a["y"] += a["vy"]

            # Borda elástica para manter no campo
            if a["x"] < 50 or a["x"] > width - 50:
                a["vx"] *= -1
            if a["y"] < 80 or a["y"] > height - 80:
                a["vy"] *= -1

            cx, cy = int(a["x"]), int(a["y"])
            ax_len = a["len"]
            ax_w = a["width"]

            # Ângulo de rotação baseado na velocidade
            angle = int(np.degrees(np.arctan2(a["vy"], a["vx"])))

            # Sombra projetada (preta translúcida deslocada para sudeste)
            shadow_center = (cx + 12, cy + 14)
            cv2.ellipse(
                frame,
                shadow_center,
                (ax_len + 4, ax_w + 2),
                angle,
                0,
                360,
                (20, 45, 25),
                -1,
                cv2.LINE_AA,
            )

            # Corpo do animal visto de cima (elipse alongada)
            cv2.ellipse(
                frame,
                (cx, cy),
                (ax_len, ax_w),
                angle,
                0,
                360,
                a["color"],
                -1,
                cv2.LINE_AA,
            )

            # Cabeça do animal
            head_offset = int(ax_len * 0.8)
            rad = np.radians(angle)
            hx = int(cx + head_offset * np.cos(rad))
            hy = int(cy + head_offset * np.sin(rad))
            cv2.circle(
                frame,
                (hx, hy),
                int(ax_w * 0.7),
                a["color"],
                -1,
                cv2.LINE_AA,
            )

        out.write(frame)

    out.release()
    print(f"[Gerador] Vídeo sintético salvo com sucesso em: {output_path}")


if __name__ == "__main__":
    create_synthetic_drone_video()
