import os
import numpy as np
import cv2
import tempfile
from app.detect import detect_players
from app.tracker import PlayerTracker
from app.classify import classify_players_teams
from app.annotate import annotate_players

if not hasattr(np, 'float'):
    np.float = float

def run_pipeline(
    input_video_bytes: bytes,
    video_input_path = "../videos/short.mp4",
):
    input_tempfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_tempfile.write(input_video_bytes)
    input_tempfile.close()
    input_tempfile_path = input_tempfile.name

    cap = cv2.VideoCapture(input_tempfile_path)
    tracker = PlayerTracker()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_tempfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_tempfile.close()
    output_tempfile_path = output_tempfile.name

    out = cv2.VideoWriter(output_tempfile_path, fourcc, fps, (width, height))
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No frame captured")
            break
        frame_counter+=1
        if frame_counter % 10 == 0:
            continue

        detections = detect_players(frame)
        tracker.update(detections, (frame.shape[0], frame.shape[1]))
        classify_players_teams(frame, tracker)
        annotated_frame = annotate_players(frame, tracker)

        out.write(annotated_frame)

    cap.release()
    out.release()

    with open(output_tempfile_path, 'rb') as f:
        processed_video_bytes = f.read()
    os.remove(output_tempfile_path)
    return processed_video_bytes

if __name__ == "__main__":
    #video_path = input("Enter path to your .mp4 video file: ").strip()
    video_path = r"D:\ACM\Pitch2Pixels\videos\short.mp4"
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    
    processed_video = run_pipeline(video_bytes)

    output_path = r"D:\ACM\Pitch2Pixels\videos\output.mp4"
    with open(output_path, "wb") as f:
        f.write(processed_video)