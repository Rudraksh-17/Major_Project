from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")


def detect_traffic(video_path):

    cap = cv2.VideoCapture(video_path)

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Save processed output
    output_path = "output_traffic.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    total_vehicles = 0
    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Run YOLO
        results = model(frame)

        vehicle_count = 0

        annotated_frame = frame.copy()

        for r in results:

            boxes = r.boxes

            for box in boxes:

                cls = int(box.cls[0])

                # Vehicle classes
                if cls in [2, 3, 5, 7]:

                    vehicle_count += 1

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

                    # Put label
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

        # Write processed frame
        out.write(annotated_frame)

    cap.release()
    out.release()

    avg_vehicles = total_vehicles / max(frame_count, 1)

    traffic_density = min(avg_vehicles / 20, 1.0)

    return avg_vehicles, traffic_density, output_path
