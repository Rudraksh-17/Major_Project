from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")


def detect_traffic(video_path):

    cap = cv2.VideoCapture(video_path)

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Output video
    output_path = "output_traffic.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

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

        annotated_frame = frame.copy()

        for r in results:

            for box in r.boxes:

                cls = int(box.cls[0])

                # COCO vehicle classes
                if cls in [2, 3, 5, 7]:

                    vehicle_count += 1

                    # Bounding box coordinates
                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    conf = float(box.conf[0])

                    label = f"{model.names[cls]} {conf:.2f}"

                    # Draw rectangle
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 255),
                        2
                    )

                    # Draw label
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2
                    )

        total_vehicles += vehicle_count

        # Save annotated frame
        out.write(annotated_frame)

    cap.release()
    out.release()

    avg_vehicles = total_vehicles / max(processed_frames, 1)

    # Density calculation
    traffic_density = min(avg_vehicles / 20, 1.0)

    return avg_vehicles, traffic_density, output_path
