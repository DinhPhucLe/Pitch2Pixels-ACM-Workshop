import numpy as np
import cv2

def annotate_players(
    frame: np.ndarray,
    tracker
):
    dist = lambda x, y : abs(x-y)
    current_track = tracker.current_track
    track_to_hue, track_to_team = tracker.track_to_hue, tracker.track_to_team
    centroid0, centroid1 = tracker.centroid0, tracker.centroid1

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
    return frame