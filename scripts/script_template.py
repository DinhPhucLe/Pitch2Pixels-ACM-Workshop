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

    output_tempfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_tempfile.close()
    output_tempfile_path = output_tempfile.name
    
    ########## COMPLETE THE FUNCTION ##########
    
    ### PREPARE TO WRITE VIDEO OUTPUT ###
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # out = cv2.VideoWriter(output_tempfile_path, fourcc, fps, (width, height))

    ###########################################

    with open(output_tempfile_path, 'rb') as f:
        processed_video_bytes = f.read()
    os.remove(output_tempfile_path)
    return processed_video_bytes

if __name__ == "__main__":
    #D:\ACM\Pitch2Pixels\videos\short.mp4
    video_path = r"D:\ACM\Pitch2Pixels\videos\short.mp4"
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    
    processed_video = run_pipeline(video_bytes)

    #D:\ACM\Pitch2Pixels\videos\output.mp4
    output_path = r"D:\ACM\Pitch2Pixels\videos\output.mp4"
    with open(output_path, "wb") as f:
        f.write(processed_video)