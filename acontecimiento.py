#!/usr/bin/env python3
"""
Existir como acontecimiento
============================
Obra interactiva / Visión por Computadora (OpenCV)

Concepto:
La realidad no es reconocida como un cuerpo, anatomía o identidad,
sino exclusivamente como perturbación y cambio en el espacio.
En quietud absoluta, la existencia se disuelve por completo.

Controles:
  [ESC] o [q] : Salir
  [+] / [-]   : Aumentar / Disminuir sensibilidad de detección
  [d] / [f]   : Aumentar / Disminuir velocidad de desvanecimiento (damping)
  [c]         : Alternar paleta cromática (Acuática profunda / Ámbar otoñal / Térmica espectral)
  [r]         : Resetear acumulador de perturbación
"""

import cv2
import numpy as np
import time

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo acceder a la cámara web.")
        return

    proc_w, proc_h = 320, 240
    threshold_val = 26
    damping = 0.90
    color_mode = 0
    
    prev_gray = None
    disturbance_buffer = np.zeros((proc_h, proc_w), dtype=np.float32)

    fps_time = time.time()
    fps_counter = 0
    current_fps = 0.0

    print("=============================================================")
    print("EXISTIR COMO ACONTECIMIENTO // OpenCV")
    print("La máquina no recuerda tu forma; solo recuerda tus cambios.")
    print("Presiona ESC o 'q' para salir.")
    print("=============================================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ALERTA] Cuadro de cámara no disponible.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        small = cv2.resize(frame, (proc_w, proc_h))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        if prev_gray is None:
            prev_gray = gray_blurred
            continue

        # 1. Diferenciación de cuadros (Frame differencing)
        diff = cv2.absdiff(gray_blurred, prev_gray)
        prev_gray = gray_blurred

        # 2. Filtrado y umbral de movimiento (destellos en bordes de cambio)
        _, motion_mask = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)
        motion_norm = (motion_mask.astype(np.float32) / 255.0)

        # 3. Propagación y difusión de torbellinos de tinta química / calor
        disturbance_buffer = disturbance_buffer * damping + motion_norm * 0.32
        disturbance_buffer = cv2.GaussianBlur(disturbance_buffer, (5, 5), 0)
        np.clip(disturbance_buffer, 0.0, 1.0, out=disturbance_buffer)

        # 4. Generación de la visualización artística cromática
        display_map = cv2.resize(disturbance_buffer, (w, h), interpolation=cv2.INTER_CUBIC)
        norm_intensity = (display_map * 255.0).astype(np.uint8)

        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:, :] = [16, 12, 8]  # Tono base azul turba / estanque profundo (BGR)

        if color_mode == 0:
            b = np.clip(display_map * 255.0, 0, 255).astype(np.uint8)
            g = np.clip(display_map * 230.0, 0, 255).astype(np.uint8)
            r = np.clip(display_map * 65.0, 0, 255).astype(np.uint8)
        elif color_mode == 1:
            b = np.clip(display_map * 40.0, 0, 255).astype(np.uint8)
            g = np.clip(display_map * 160.0, 0, 255).astype(np.uint8)
            r = np.clip(display_map * 255.0, 0, 255).astype(np.uint8)
        else:
            colored = cv2.applyColorMap(norm_intensity, cv2.COLORMAP_OCEAN)
            r, g, b = colored[:, :, 2], colored[:, :, 1], colored[:, :, 0]

        perturbation_layer = np.stack([b, g, r], axis=-1)
        alpha = np.repeat(display_map[:, :, np.newaxis], 3, axis=-1)
        final_view = (canvas * (1.0 - alpha) + perturbation_layer * alpha).astype(np.uint8)

        # 5. Medición de FPS
        fps_counter += 1
        now = time.time()
        if now - fps_time >= 1.0:
            current_fps = fps_counter / (now - fps_time)
            fps_counter = 0
            fps_time = now

        # Overlay tipográfico
        cv2.putText(final_view, "EXISTIR COMO ACONTECIMIENTO", (24, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 245, 250), 1, cv2.LINE_AA)
        cv2.putText(final_view, "La quietud disuelve la forma. Solo el cambio existe.", (24, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 165, 175), 1, cv2.LINE_AA)
        cv2.putText(final_view, f"FPS: {current_fps:.1f} | Sensibilidad: {threshold_val} | Inercia: {damping:.2f}",
                    (24, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 140, 150), 1, cv2.LINE_AA)

        cv2.imshow("Existir como acontecimiento - OpenCV", final_view)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            threshold_val = max(5, threshold_val - 2)
            print(f"[AJUSTE] Sensibilidad aumentada (Umbral: {threshold_val})")
        elif key == ord('-') or key == ord('_'):
            threshold_val = min(50, threshold_val + 2)
            print(f"[AJUSTE] Sensibilidad disminuida (Umbral: {threshold_val})")
        elif key == ord('d'):
            damping = min(0.98, damping + 0.01)
            print(f"[AJUSTE] Inercia aumentada: {damping:.2f}")
        elif key == ord('f'):
            damping = max(0.70, damping - 0.01)
            print(f"[AJUSTE] Inercia disminuida: {damping:.2f}")
        elif key == ord('c'):
            color_mode = (color_mode + 1) % 3
            names = ["Acuatica Profunda", "Ambar Otonal", "Espectral Oceanica"]
            print(f"[AJUSTE] Paleta: {names[color_mode]}")
        elif key == ord('r'):
            disturbance_buffer.fill(0)
            print("[AJUSTE] Buffer reseteado.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
