import cv2
import numpy as np
from sklearn.cluster import KMeans

def getDominantColor(frame):
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

def classify_players_teams(
    frame,
    tracker
):
    current_track = tracker.current_track

    for track in current_track:
        x, y, w, h = map(int, track.tlwh)
        track_id = track.track_id

        if track_id not in tracker.track_to_hue:
            crop = frame[int(y+h*0.25):int(y+h*0.6), int(x+w*0.25):int(x+w*0.75)]
            dom_hue = getDominantColor(crop)
            tracker.track_to_hue[track_id] = dom_hue
    
    if len(tracker.track_to_hue) >= 2 and tracker.clustered == False:
        tracker.track_to_team.update(cluster_players_by_color(tracker.track_to_hue))
        hue0 = [hue for track_id, hue in tracker.track_to_hue.items() if tracker.track_to_team[track_id] == 0]
        hue1 = [hue for track_id, hue in tracker.track_to_hue.items() if tracker.track_to_team[track_id] == 1]
        tracker.centroid0 = np.mean(hue0)
        tracker.centroid1 = np.mean(hue1)
        tracker.clustered = True
    return
    