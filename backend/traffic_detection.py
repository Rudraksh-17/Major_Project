# =========================
# traffic_detection.py
# =========================

from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")


def detect_traffic(video_path):

    cap = cv2.VideoCapture(video_path)

    total_vehicles = 0
    processed_frames = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        processed_frames += 1

        # Skip frames for speed
        if processed_frames % 10 != 0:
            continue

        results = model(frame)

        vehicle_count = 0

        for r in results:

            for box in r.boxes:

                cls = int(box.cls[0])

                # Vehicle classes
                # car=2, motorcycle=3, bus=5, truck=7
                if cls in [2, 3, 5, 7]:
                    vehicle_count += 1

        total_vehicles += vehicle_count

    cap.release()

    avg_vehicles = total_vehicles / max(processed_frames, 1)

    # Traffic density normalization
    traffic_density = min(avg_vehicles / 20, 1.0)

    return int(avg_vehicles), float(traffic_density)
