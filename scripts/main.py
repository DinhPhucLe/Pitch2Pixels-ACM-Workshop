import cv2
from ultralytics import YOLO
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker
from sklearn.cluster import KMeans
import argparse

track_to_hue = {}
track_to_team = {}
clustered = False
centroid0, centroid1 = 0, 0

def dist(x, y):
    return abs(x-y)

def getDominantColor(frame, track_id):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat_mask = cv2.inRange(hsv[:, :, 1], 60, 255)
    val_mask = cv2.inRange(hsv[:, :, 2], 60, 255)
    colorful_mask = cv2.bitwise_and(sat_mask, val_mask)

    mask1 = cv2.inRange(hsv, (0, 60, 60), (34, 255, 255))   # reds to yellows
    mask2 = cv2.inRange(hsv, (86, 60, 60), (180, 255, 255)) # cyans to pinks
    hue_mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.bitwise_and(colorful_mask, hue_mask)
    masked_hue = hsv[:, :, 0][mask > 0]

    hist = cv2.calcHist([masked_hue], [0], None, [180], [0, 180])

    dominant_hue = int(np.argmax(hist))
    return dominant_hue

def cluster_players_by_color(dominant_hues):
    data = np.array(list(dominant_hues.values())).reshape(-1, 1)
    kmeans = KMeans(n_clusters = 2, random_state = 42, n_init = "auto")
    team_labels = kmeans.fit_predict(data)
    return dict(zip(dominant_hues.keys(), team_labels))

if not hasattr(np, 'float'):
    np.float = float

args = argparse.Namespace(
    track_thresh=0.5,
    track_buffer=30,
    match_thresh=0.8,
    mot20=False,
    min_box_area=10,
    aspect_ratio_thresh=1.6
)

tracker=BYTETracker(
    args,
    frame_rate=30
)

model = YOLO("../model/best-yolov8s.pt")
cap = cv2.VideoCapture("../videos/ars_vs_mci.mp4")
frame_counter = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('../videos/output.mp4', fourcc, 20.0, (1920, 1080))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No frame captured")
        break
    frame_counter += 1

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

    if len(detections) == 0:
        track_to_team.clear()
        track_to_hue.clear()
        continue

    current_track = []
    img_info = (frame.shape[0], frame.shape[1])
    current_track = tracker.update(detections, img_info, img_info)

    for track in current_track:
        x, y, w, h = map(int, track.tlwh)
        track_id = track.track_id
        
        if track_id not in track_to_hue:
            crop = frame[int(y+h*0.25):int(y+h*0.6), int(x+w*0.25):int(x+w*0.75)]
            dom_hue = getDominantColor(crop, track_id)
            track_to_hue[track_id] = dom_hue

    if len(track_to_hue) >= 2 and not clustered:
        track_to_team.update(cluster_players_by_color(track_to_hue))
        hue0 = [hue for track_id, hue in track_to_hue.items() if track_to_team[track_id] == 0]
        hue1 = [hue for track_id, hue in track_to_hue.items() if track_to_team[track_id] == 1]
        centroid0 = np.mean(hue0) # Take the mean of all heu that is labeled 0
        centroid1 = np.mean(hue1) # Take the mean of all hue that is labeled 1
        clustered = True

    for track in current_track:
        x, y, w, h = map(int, track.tlwh)
        track_id = track.track_id
        hue = track_to_hue[track_id]
        if dist(hue, centroid1) > 30 and dist(hue, centroid0) > 30:
            continue
        team = 1 if dist(hue, centroid1) < dist(hue, centroid0) else 0
        color = (255, 0, 255) if team == 0 else (255, 215, 0)

        cv2.ellipse(
            img = frame,
            center = (int(x+w/2), int(y+h)),
            axes = (int(w), int(h/6)),
            angle = 0,
            startAngle = -30,
            endAngle = 210,
            color = color,
            thickness = 2
        )
        cv2.putText(frame, f"#{track_id} T{team+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
    #out.write(frame)
    cv2.imshow("tracking", frame)
    print(track_to_hue)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(frame_counter)

# Check frame 750+++