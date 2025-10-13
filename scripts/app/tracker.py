from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
import argparse

class PlayerTracker:
    def __init__(self):
        self.current_track = []
        self.tracker = BYTETracker(
            args = argparse.Namespace(
                track_thresh=0.5,
                track_buffer=30,
                match_thresh=0.8,
                mot20=False,
                min_box_area=10,
                aspect_ratio_thresh=1.6
            ),
            frame_rate = 30
        )

        self.track_to_team = {}
        self.track_to_hue = {}

        self.centroid0 = None
        self.centroid1 = None
        self.clustered = False
    
    def update(self, detections, img_info):
        self.current_track = self.tracker.update(detections, img_info, img_info)
        return self