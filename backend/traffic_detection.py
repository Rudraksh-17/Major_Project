from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

vehicle_classes = ['car', 'truck', 'bus', 'motorbike']


def detect_traffic(video_path):

    cap = cv2.VideoCapture(video_path)

    vehicle_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)

        detections = results[0]

        count = 0

        for box in detections.boxes:

            cls_id = int(box.cls[0])

            class_name = model.names[cls_id]

            if class_name in vehicle_classes:
                count += 1

        vehicle_count = count

        break

    cap.release()

    # Convert count to density
    if vehicle_count < 5:
        density = 0.2

    elif vehicle_count < 15:
        density = 0.5

    else:
        density = 0.9

    return vehicle_count, density