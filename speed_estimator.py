import os

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import numpy as np
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR
from base_solution import BaseSolution
import torch
import torch.nn as nn
import torch.nn.functional as F


import nltk
from nltk.corpus import words

try:
    EN_WORDS = set(w.upper() for w in words.words())
except LookupError:
    nltk.download("words")
    EN_WORDS = set(w.upper() for w in words.words())

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.rnn = nn.LSTM(128, nh, 2, bidirectional=True)
        self.fc = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        return x


class SpeedEstimator(BaseSolution):
    def __init__(self, model, region, line_width=2, ocr_interval=5):
        super().__init__(model=model, region=region, line_width=line_width)

        self.detector = YOLO(model).to(DEVICE)

        self.ocr = PaddleOCR(
            lang="en",
            use_angle_cls=True,
            use_gpu=False,
            show_log=False
        )

        self.ocr_interval = ocr_interval
        self.frame_count = 0
        self.last_text = {}
        self.last_centers = {}
        self.next_plate_id = 0

    
    def filter_plate_text(self, ocr_result, plate_h):
        if not ocr_result or not ocr_result[0]:
            return ""

        lines = []

        
        for det in ocr_result[0]:
            if len(det) != 2:
                continue

            box = np.array(det[0])
            text, score = det[1]

            if score < 0.4:
                continue

            
            text = (
                text.upper()
                .replace(" ", "")
                .replace(".", "")
                .replace("_", "-")
                .replace("—", "-")
            )
            text = re.sub(r"[^A-Z0-9\-]", "", text)

            if len(text) < 2 or len(text) > 12:
                continue

            
            if text in EN_WORDS:
                continue

            
            y_coords = box[:, 1]
            line_h = y_coords.max() - y_coords.min()
            center_y = y_coords.mean()

            
            if line_h < plate_h * 0.18:
                continue

            lines.append((center_y, text))

        if not lines:
            return ""

        
        lines.sort(key=lambda x: x[0])
        grouped = []

        for y, txt in lines:
            placed = False
            for group in grouped:
                if abs(group[0] - y) < plate_h * 0.15:
                    group[1].append(txt)
                    placed = True
                    break
            if not placed:
                grouped.append([y, [txt]])

        
        grouped = grouped[:2]

        
        final_text = ""
        for _, texts in grouped:
            final_text += "".join(texts)

        
        if 5 <= len(final_text) <= 12:
            return final_text

        return ""


    
    def perform_ocr(self, plate):
        try:
            result = self.ocr.ocr(plate)
            return self.filter_plate_text(result, plate.shape[0])
        except Exception as e:
            print("OCR ERROR:", e)
            return ""

    
    def match_plate(self, box, threshold=50):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        for pid, (px, py) in self.last_centers.items():
            if abs(cx - px) < threshold and abs(cy - py) < threshold:
                return pid
        pid = self.next_plate_id
        self.next_plate_id += 1
        return pid

    
    def _detect_frame_core(self, frame, force_ocr):
        output = frame.copy()
        h, w = frame.shape[:2]
        self.frame_count += 1
        current_centers = {}
        plate_detected = False

        results = self.detector.predict(frame, imgsz=640, conf=0.3, iou=0.6, verbose=False)
        if not results or results[0].boxes is None:
            return output, plate_detected

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if self.detector.names[cls_id].lower() != "number_plate":
                continue

            plate_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate = frame[y1:y2, x1:x2]
            if plate.size == 0:
                continue

            pid = self.match_plate((x1, y1, x2, y2))
            current_centers[pid] = ((x1 + x2)//2, (y1 + y2)//2)

            if force_ocr or self.frame_count % self.ocr_interval == 0:
                text = self.perform_ocr(plate)
                if text:
                    self.last_text[pid] = text

            final_text = self.last_text.get(pid, "")

            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if final_text:
                cv2.putText(output, final_text, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        self.last_centers = current_centers
        return output, plate_detected

    
    def detect_frame_np(self, frame, force_ocr=False, max_dim=800):
        h0, w0 = frame.shape[:2]
        scale = min(max_dim / max(h0, w0), 1.0)
        small = cv2.resize(frame, (int(w0*scale), int(h0*scale))) if scale < 1 else frame
        out_small, plate_detected = self._detect_frame_core(small, force_ocr)
        out = cv2.resize(out_small, (w0, h0)) if scale < 1 else out_small
        return out, plate_detected

    
    def detect_frame(self, frame, force_ocr=True, max_dim=800):
        out, plate_detected = self.detect_frame_np(frame, force_ocr, max_dim)
        if not plate_detected:
            return None
        _, buf = cv2.imencode(".jpg", out)
        return buf.tobytes()



speed_obj = SpeedEstimator(
    model="best_finetuned.pt",
    region=[(0, 145), (1018, 145)],
    line_width=2,
    ocr_interval=5
)


_dummy = np.zeros((200, 400, 3), dtype=np.uint8)
try:
    speed_obj.detector.predict(_dummy, verbose=False)
    speed_obj.ocr.ocr(_dummy)
except:
    pass
