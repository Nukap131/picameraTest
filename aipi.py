from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime

# Load IMX500 model (person‑detection)
# Brug f.eks. MobileNet SSD v2 320x320 model fra /usr/share/imx500-models
model_path = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
imx500 = IMX500(model_path)

# Initialiser kamera og konfiguration
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)

# IMX500 sørger for at tilføje AI‑output‑bufferen
imx500.show_preview(True)
picam2.post_process = [imx500]

picam2.start()

frame_width, frame_height = 640, 480
line_x = frame_width // 2
line_y1, line_y2 = 0, frame_height

total_crossings = 0
cross_history = defaultdict(list)  # ID -> liste af x‑positioner
cross_log = []
last_cross_time = {}
cooldown_seconds = 1

print(f"Startet kl. {datetime.now().strftime('%d-%m-%y %H:%M:%S')}")

while True:
    frame = picam2.capture_array()

    # IMX500 returnerer objektdetektioner i imx500.results
    ai_results = imx500.get_results()
    boxes = ai_results.bbox  # Normaliserede koordinater: [x,y,w,h] mellem 0–1
    labels = ai_results.label
    scores = ai_results.conf

    for i in range(len(labels)):
        score = scores[i]
        if score < 0.5:
            continue

        # Kun “person” (klassenummer 0 i mange SSD‑modeller)
        # Tjek evt. labels i din model (brug label‑navne eller ID‑48 for person i COCO)
        if labels[i] != 0:  # eller hvad “person” er i din SSD‑model
            continue

        # Fra normaliseret [0..1] til pixel‑koordinater
        x_norm, y_norm, w_norm, h_norm = boxes[i]
        x1 = int(x_norm * frame_width)
        y1 = int(y_norm * frame_height)
        x2 = int((x_norm + w_norm) * frame_width)
        y2 = int((y_norm + h_norm) * frame_height)

        cx = (x1 + x2) // 2
        track_id = i  # eller brug en mere avanceret tracker ovenpå

        cross_history[track_id].append(cx)
        if len(cross_history[track_id]) > 10:
            cross_history[track_id].pop(0)

        if len(cross_history[track_id]) > 1:
            prev_cx = cross_history[track_id][-2]
            now = datetime.now()
            last_time = last_cross_time.get(track_id, None)
            too_soon = last_time and (now - last_time).total_seconds() < cooldown_seconds

            if not too_soon:
                if (prev_cx < line_x and cx >= line_x) or (prev_cx > line_x and cx <= line_x):
                    direction = "→" if prev_cx < line_x else "←"
                    last_cross_time[track_id] = now

                    # Tæl kun “højre → venstre”
                    if prev_cx > line_x and cx <= line_x:
                        total_crossings += 1

                    ts = now.strftime('%d-%m-%y %H:%M:%S')
                    cross_log.append(f"{ts} | ID {track_id} | {direction} | Total: {total_crossings}")
                    print(f"{ts} - ID {track_id} krydsede {direction}! Total: {total_crossings}")

        # Tegn bounding box og ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # Tegn linje og info
    cv2.line(frame, (line_x, line_y1), (line_x, line_y2), (0, 0, 255), 3)
    cv2.putText(frame, f"TOTAL: {total_crossings}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Tid: {datetime.now().strftime('%H:%M:%S')}",
                (10, frame_height - 2 популярно neс,L,-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Person Tæller - Pi5 + IMX500", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

picam2.close()
cv2.destroyAllWindows()

# Gem logfil
filename = f"krydsninger_{datetime.now().strftime('%d%m%y_%H%M%S')}.txt"
with open(filename, "w", encoding='utf-8') as f:
    f.write(f"=== PERSON TÆLLER LOG (IMX500) ===\n")
    f.write(f"Total personer, der går ind: {total_crossings}\n")
    for log_entry in cross_log:
        f.write(log_entry + "\n")

print(f"Færdig! {total_crossings} krydsninger gemt i {filename}")
