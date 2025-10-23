# Pitch2Pixels: Player Detection, Tracking, and Team Classification

This workshop demonstrates how to detect, track, and classify football players from match footage using **YOLO**, **BYTETracker**, and color-based clustering techniques. Participants will learn how to extract positional data from videos and visualize it for sports analytics applications.

---

## 🚀 PROJECT WORKFLOW

1. **Detection**  
   Use **YOLO** (You Only Look Once) to detect football players frame by frame.

2. **Tracking**  
   Apply **BYTETracker** to assign consistent IDs to each detected player across frames.

3. **Team Classification**  
   - Extract **HSV color features** from each player's bounding box.  
   - Use **K-Means clustering** to group players into two teams based on uniform colors.

4. **Visualization & Analysis**  
   Overlay bounding boxes, IDs, and team colors on the video to visualize tracking results and team separation.

---

## 🧠 TECH STACKS

- **Python 3.8+**
- **YOLOv8** (via [Ultralytics](https://github.com/ultralytics/ultralytics))
- **BYTETrack** (for multi-object tracking)
- **OpenCV** (for video processing and HSV extraction)
- **scikit-learn** (for K-Means clustering)

---

## ⚙️ HOW TO SET UP

1. **Clone the repository**
   ```bash
   git clone https://github.com/DinhPhucLe/Pitch2Pixels-ACM-Workshop.git
   cd Pitch2Pixels-ACM-Workshop
   ```

2. **Create and activate a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your input video**
   - Place your football match footage in your chosen path.
   - Update paths in the script accordingly.

5. **Run the notebook or script**
   From root terminal, run:
   ```bash
   python scripts/main.py
   ```

6. **Visualize the output**
   - Update the output path to your preferences in `scripts/script.py`
   - The processed video with detection, tracking, and team classification overlays will be saved in your designated path.

---

## 📚 REFERENCES

- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)  
- [BYTETrack: Multi-Object Tracking](https://github.com/ifzhang/ByteTrack)  
- [OpenCV Documentation](https://docs.opencv.org/)  
- [scikit-learn K-Means Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)  
- [Dataset & Testing Videos](https://drive.google.com/drive/folders/1MvQuZFKwzTN1bDBAIkh_-bxvpqRK6vhI?usp=sharing)

