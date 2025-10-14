# ⚽ Pitch2Pixels: Computer Vision for Football Tactics ⚽
A hands-on computer vision workshop showcasing football player detection and tracking using YOLOv8 and BYTETracker, with real-time clustering and team identification.
## ✨ Features
* **Real-time Player Detection:** Utilizes YOLOv8 for accurate and efficient detection of football players in video streams.
* **Robust Player Tracking:** Employs BYTETracker to maintain consistent player IDs across frames, even during occlusions.
* **Automated Team Identification:** Automatically assigns players to teams based on dominant jersey color using KMeans clustering.
* **Annotated Video Output:** Generates output videos with bounding boxes, player IDs, and team labels.
## 🚀 Technologies
* **YOLOv8 (Ultralytics)** for players detection
* **BYTETracker** for players tracking
* **OpenCV** for video processing and color analysis
* **KMeans Clustering** for jersey color clustering
## ⚙️ Pipeline
1. Video Input 📹
2. YOLOv8 detects players per frame 🕵️
3. BYTETracker tracks and assigns players ID
4. For each new track 🖼️
   - Crop player's jersey area
   - Extract dominant color in HSV
   - Append to hue list
5. KMeans clusters players into teams (only first frame)
6. Assign team labels to players 🏷️
7. Output annotated video
## 📌 How to Run?
Set up your Virtual Environment
```bash
python -m venv .venv
.venv/Scripts/activate
```
Install required libraries and packages
```bash
pip install -r requirements.txt
```
Go to `/scripts/pipeline.py` and edit the `video_path` to where you store your footage and `output_path` to your preferences.
```bash
...
if __name__ == "__main__":
    video_path = r"WHERE_YOU_STORE_YOUR_VIDEO"
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    
    processed_video = run_pipeline(video_bytes)

    output_path = r"YOUR_OUTPUT_PATH"
    with open(output_path, "wb") as f:
        f.write(processed_video)
...
```
