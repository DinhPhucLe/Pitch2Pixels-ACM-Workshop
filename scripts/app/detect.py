from ultralytics import YOLO
import numpy as np

def detect_players(
    frame,
    model = YOLO("scripts/model/best-yolov8s.pt")
) -> np.ndarray:
    results = model(frame, conf=0.4, iou=0.5)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    conf = results.boxes.conf.cpu().numpy()
    cls = results.boxes.cls.cpu().numpy()

    detections = []
    for box, conf, cls in zip(boxes, conf, cls):
        if int(cls) != 2:
            continue
        x1, y1, x2, y2 = box
        detections.append([x1, y1, x2, y2, conf])
    detections = np.array(detections)
    return detections