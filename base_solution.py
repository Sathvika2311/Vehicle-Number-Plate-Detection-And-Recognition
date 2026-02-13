

import cv2
import numpy as np
from shapely.geometry import LineString
from ultralytics import YOLO


class BaseSolution:
    def __init__(self, model, region, line_width=2, **kwargs):
        self.model = YOLO(model)
        self.region = region
        self.line_width = line_width
        self.track_ids = []
        self.boxes = []
        self.clss = []
        self.names = self.model.model.names
        self.track_line = []
        self.r_s = LineString(region)

    def initialize_region(self):
        self.track_line = [tuple(p) for p in self.region]

    def extract_tracks(self, im0):
        results = self.model.track(im0, persist=True, verbose=False)
        result = results[0]
        if result and result.boxes and result.boxes.id is not None:
            self.boxes = result.boxes.xyxy.cpu().numpy()
            self.track_ids = result.boxes.id.int().cpu().tolist()
            self.clss = result.boxes.cls.int().cpu().tolist()
        else:
            self.boxes = []
            self.track_ids = []
            self.clss = []

    def store_tracking_history(self, track_id, point):
        
        pass

    def display_output(self, im0):
        
        pass
