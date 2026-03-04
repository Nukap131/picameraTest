from picamera2 import Picamera2
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime
import time
import os

# SLÅ ALT YOLO GUI FRA FØR import
os.environ["YOLO_VERBOSE"] = "False"
os.environ["ULTRALYTICS_SHOW"] = "False"

print(" FABLAB PERSON TÆLLER - Pi AI Camera")
print("Tryk 'q' for at stoppe")

# Kamera setup
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(2)

frame_width, frame_height = 640, 480
line_x = 320  # Midten af billedet (tæller kun højre→venstre)

# YOLO - BARE RAW DATA, INGEN VISUALS!
model = YOLO("yolov8n.pt")
model.overrides['verbose'] = False
model.overrides['show'] = False
model.overrides['save'] = False
model.overrides['visualize'] = False

# Counters
total_crossings = 0
cross_history = defaultdict(list)
cross_log = []
last_cross_time = {}
cooldown_seconds = 1

print(f"Startet: {datetime.now().strftime('%d-%m-%y %H:%M:%S')}")
print(f"Linje: x={line_x} (tæller venstre→højre)")


try:
    while True:
        # RAW frame
        frame = picam2.capture_array()
        
        # Fix IMX500 color
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # SPEJL-FIX
        frame = cv2.flip(frame, 1)
        
        # YOLO PREDICTION
        results = model.track(
            frame, 
            persist=True, 
            classes=[0],
            conf=0.5,
            tracker="bytetrack.yaml",
            verbose=False,
            show=False,
            save=False,
            stream=False,
            project="", 
            name=""
        )

        if results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2

                cross_history[track_id].append(cx)
                if len(cross_history[track_id]) > 10:
                    cross_history[track_id].pop(0)

                if len(cross_history[track_id]) > 1:
                    prev_cx = cross_history[track_id][-2]
                    now = datetime.now()

                    # Tæller kun højre → venstre (←) - samme logformat som før
                    if prev_cx > line_x and cx <= line_x:
                        direction = "←"

                        last_time = last_cross_time.get(track_id)
                        too_soon = last_time and (now - last_time).total_seconds() < cooldown_seconds

                        if not too_soon:
                            last_cross_time[track_id] = now
                            total_crossings += 1

                            timestamp = now.strftime('%d-%m-%y %H:%M:%S')
                            log_entry = f"{timestamp} | ID{track_id} {direction} | Total: {total_crossings}"
                            cross_log.append(log_entry)
                            print(f" {log_entry}")

                    # `→` (venstre → højre) tælles IKKE, men kan stadig logges, hvis du vil
                    # (fjern hele blokken hvis du ikke vil logge `→`-passager)
                    elif prev_cx < line_x and cx >= line_x:
                        direction = "→"
                        timestamp = now.strftime('%d-%m-%y %H:%M:%S')
                        log_entry = f"{timestamp} | ID{track_id} {direction} | Total: {total_crossings}"
                        cross_log.append(log_entry)
                        print(f" {log_entry}")

                # Tegn boks
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Tegn linje og tekst
        cv2.line(frame, (line_x, 0), (line_x, frame_height), (0, 0, 255), 3)
        cv2.putText(frame, f"TOTAL ind: {total_crossings}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Tracks: {len(cross_history)}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Tid: {datetime.now().strftime('%H:%M:%S')}", 
                    (10, frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("FABLAB Person Taeller", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n Stoppet")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    
    # SAMME LOGFIL FORMAT SOM DU VIL HA'
    filename = f"fablab_log_{datetime.now().strftime('%d%m%y_%H%M%S')}.txt"
    with open(filename, "w", encoding='utf-8') as f:
        f.write("===FABLAB PERSON TÆLLER===\n")
        f.write(f"TOTAL personer, der går ind i Fablab: {total_crossings}\n")
        f.write(f"Linje x={line_x}\n")
        f.write(f"Events: {len(cross_log)}\n\n")
        for entry in cross_log[-50:]:  # kan evt. være alle hvis du vil: cross_log
            f.write(entry + "\n")
    
    print(f"\n FÆRDIG! {total_crossings} talt. Log: {filename}")
