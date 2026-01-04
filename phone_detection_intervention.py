#!/usr/bin/env python
"""
PHONE / TABLET OBJECT-DETECTION + YOUTUBE INTERVENTION
- Uses YOLOv8 (COCO: cell phone 67, laptop 63, tv 62)
- Draws green bounding boxes around detected phones / screens
- Triggers as soon as ANY device is seen in a frame (with cooldown)
"""

import sys
import time
import random
import os
import platform

import cv2
from ultralytics import YOLO  # pip install ultralytics

print("=" * 70)
print("YOLO-BASED PHONE / TABLET DETECTION SYSTEM (INSTANT ON 'DEVICE SEEN')")
print("=" * 70)

# ==================== YOUTUBE VIDEO LIST ====================

INTERVENTION_VIDEOS = [
    # Phone Addiction & Digital Detox
    "https://www.youtube.com/watch?v=Idh24Mc72mg",
    "https://www.youtube.com/watch?v=5ZonuuFa1no",
    "https://www.youtube.com/watch?v=uOFGswohHAw",
    "https://www.youtube.com/watch?v=M7NDaiA4X_g",
    "https://www.youtube.com/watch?v=oUoxlF_s7wk",
    "https://www.youtube.com/watch?v=KN7RBUZL3B8",
    "https://www.youtube.com/watch?v=8cOvdrhN38o",
    "https://www.youtube.com/watch?v=giEzyRLLd1c",

    # Focus & Productivity
    "https://www.youtube.com/shorts/rVbo3RpOwKU",
    "https://www.youtube.com/shorts/MJ9qycpYr9g",
    "https://www.youtube.com/shorts/KF3etvbVN7c",
    "https://www.youtube.com/shorts/8YKQSW0vZ-c",
    "https://www.youtube.com/shorts/4dEZ6nYCWr4",

    # Dopamine & Brain Science
    "https://www.youtube.com/shorts/Lt-T-tqasUM",
    "https://www.youtube.com/shorts/s9nDVkBSr-g",
    "https://www.youtube.com/shorts/Tqjkg3Mab9g",
    "https://www.youtube.com/shorts/3O-9k5Iquxw",

    # Practical Solutions
    "https://www.youtube.com/shorts/Q4PctusE7K4",
    "https://www.youtube.com/watch?v=hSGt_rhu49U",
    "https://www.youtube.com/watch?v=ex1YT5C5Fvw",
    "https://www.youtube.com/shorts/k32vVwkxSl8",
    "https://www.youtube.com/watch?v=S39zoHnV-ok",
    "https://www.youtube.com/shorts/vhi-WpE-paA",
]

# ==================== MODEL & CAMERA SETUP ====================

print("\n[1/3] Loading YOLOv8 model (COCO pretrained)...")
try:
    # you can use yolov8n.pt or yolov8m.pt depending on what you downloaded
    model = YOLO("yolov8n.pt")
    names = model.names
    print(f"  ✅ Model loaded with {len(names)} classes")
except Exception as e:
    print(f"  ❌ Failed to load YOLOv8 model: {e}")
    sys.exit(1)

# COCO device classes: 67='cell phone', 63='laptop', 62='tv'
DEVICE_CLASS_IDS = {67, 63, 62}

print("\n[2/3] Setting up camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("  ❌ Camera failed!")
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("  ✅ Camera ready (1280x720)")

print("\n[3/3] System configuration...")
TRIGGER_COOLDOWN = 10          # seconds between interventions
BASE_CONFIDENCE = 0.25         # YOLO internal conf to get proposals
print(f"  ✅ Cooldown: {TRIGGER_COOLDOWN}s")
print("\n" + "=" * 70)
print("✅ SYSTEM READY – HOLD A PHONE IN FRONT OF THE CAMERA")
print("⌨️  Press 'q' to quit")
print("=" * 70 + "\n")

# ==================== YOUTUBE INTERVENTION ====================

last_trigger_time = 0.0
current_video_url = None

def open_youtube_video():
    """Open a phone-addiction video if cooldown passed."""
    global last_trigger_time, current_video_url

    now = time.time()
    if now - last_trigger_time < TRIGGER_COOLDOWN:
        return False

    video_url = random.choice(INTERVENTION_VIDEOS)
    current_video_url = video_url
    last_trigger_time = now

    print("\n" + "=" * 70)
    print("🚨 DEVICE SEEN – OPENING YOUTUBE INTERVENTION")
    print("🔗", video_url)
    print("=" * 70 + "\n")

    if platform.system() == "Windows":
        os.system(f'start {video_url}')
    elif platform.system() == "Darwin":
        os.system(f'open {video_url}')
    else:
        os.system(f'xdg-open {video_url}')

    return True

# ==================== MAIN LOOP ====================

frame_count = 0
device_detections = 0
interventions = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error reading frame")
            break

        frame_count += 1
        h, w, _ = frame.shape

        # Get detections with a low base confidence, filter by class only
        results = model.predict(frame, conf=BASE_CONFIDENCE, verbose=False)

        device_in_frame = False
        max_conf_this_frame = 0.0

        if results:
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    if cls_id in DEVICE_CLASS_IDS:
                        device_in_frame = True
                        device_detections += 1
                        max_conf_this_frame = max(max_conf_this_frame, conf)

                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # Always draw green box when device is seen
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        label = f"{names[cls_id]} {conf:.2f}"
                        cv2.putText(
                            frame,
                            label,
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2,
                        )

        # As soon as any device is seen, trigger (respect cooldown)
        triggered_now = False
        if device_in_frame:
            if open_youtube_video():
                triggered_now = True
                interventions += 1

        # HUD text
        cv2.putText(
            frame,
            f"Frames: {frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Device detections: {device_detections}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Interventions: {interventions}",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Max conf this frame: {max_conf_this_frame:.2f}",
            (10, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        if triggered_now:
            status = "🚨 TRIGGERED (device seen)"
            color = (0, 0, 255)
        elif device_in_frame:
            status = "📱 Device seen (cooldown active)"
            color = (0, 255, 255)
        else:
            status = "✅ No device in frame"
            color = (0, 255, 0)

        cv2.putText(
            frame,
            status,
            (10, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )
        cv2.putText(
            frame,
            "Press 'q' to quit",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        cv2.imshow("YOLO Phone Detection – INSTANT ON DEVICE SEEN – Press 'q' to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n⏹️  Stopping system...")
            break

except KeyboardInterrupt:
    print("\n⏹️  Interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\n" + "=" * 70)
    print("📊 SESSION SUMMARY")
    print("=" * 70)
    print("Frames processed:      ", frame_count)
    print("Device detections:     ", device_detections)
    print("Interventions triggered", interventions)
    print("=" * 70)
