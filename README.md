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
## 📌 Install dependencies and libraries
Run the below commands in your environment terminal to install necessary libraries and dependencies.
```bash
# Only clone the ByteTrack repo if you do NOT have it in your project folder
git clone https://github.com/ifzhang/ByteTrack.git

cd ByteTrack
pip install -r requirements.txt
pip install lap loguru cython scikit-image onnxruntime cython_bbox
python setup.py develop
pip install ultralytics
```
## 🏃 How to use?
Once the dependencies are installed, you can run the main script.
1.  **Prepare your input video:** Place your football match video (e.g., `input_match.mp4`) in a designated `videos/` folder within the project root,
or specify its path directly in the command.

2. **Adjust the video path in `/scripts/main.py` file:**
   ```bash
   ...
   model = YOLO("../model/best-yolov8s.pt")
   # Edit the input video path
   cap = cv2.VideoCapture("../videos/input.mp4")
   frame_counter = 0

   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   # Edit the desired output video name and video resolution to match input resolution
   out = cv2.VideoWriter('../videos/output.mp4', fourcc, 20.0, (resolution_width, resolution_height))
   ...
   ```

3. **Run the detection and tracking script:**
   ```bash
   cd scripts   # Change directory to the folder containing script
   python main.py   # Run the script
   ```
    * **Note:** Replace `main.py` with the actual name of your main script if it's different (e.g., `run.py`, `demo.py`).
    * The script used in this repo is `main.py`.

4. **View the output:** The annotated video will be saved as `output_match_annotated.mp4` (or whatever name you specify) in your project directory.
