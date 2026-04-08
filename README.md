# 🚗 Vehicle Number Plate Detection and Recognition Using Deep Learning

A real-time **Automatic Number Plate Detection and Recognition** system built using deep learning and computer vision. The system detects vehicle number plates from images, videos, and live streams, extracts text using CRNN, and intelligently validates and corrects Indian license plate formats.

---
## 📂 Dataset
🔗 [Click here to download dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/13/download/yolov11)

## 🤖 CRNN Model File
🔗 [Click here to download trained CRNN model](https://drive.google.com/file/d/1JM2cj7G-LRyJckutk3n9YGCW7zWj7DOc/view?usp=sharing)

---
## ✨ Key Features

- 🔍 **Accurate Plate Detection**
  - YOLOv11-based number plate detection

- 🔤 **Hybrid OCR System**
  - CRNN architecture

- 🇮🇳 **Indian Plate Validation Engine**
  - Supports Standard, BH-series, Military, Old formats
  - Regex-based validation with state verification

- 🧠 **Automatic Error Correction**
  - Fixes text errors (O→0, S→5, B→8, etc.)
  - Format-aware correction logic

- 🎥 **Multi-Input Support**
  - Image upload
  - Video processing
  - Camera capture
  - Live streaming (WebSocket)

- ⚡ **Real-Time Optimization**
  - Frame skipping
  - OCR interval control
  - Detection caching

- 📊 **Confidence Scoring**
  - Combines Detection + Recognition confidence
  - Boosting mechanism for stability

- 🧾 **History Logging**
  - Stores results with timestamp
  - Separate logs for image, video, capture, and live

---

## 🏗️ Architecture

```text
┌───────────────────────────────────────────────┐
│               Input Sources                   │
│  Image | Video | Capture | Live Stream        │
└───────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────┐
│         YOLOv11 Detection Model               │
│  • Detect number plates                       │
│  • Output bounding boxes + confidence         │
└───────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────┐
│         OCR Processing Layer                  │
│  • CRNN                                       │
│  • Multi-line text handling                   │                  
└───────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────┐
│       Text Filtering & Cleaning               │
│  • Remove noise & symbols                     │
│  • Filter invalid/English words               │
│  • Normalize text                             │
└───────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────┐
│   Indian Plate Validation Engine              │
│  • Format detection                           │
│  • Regex validation                           │
│  • Smart correction                           │
└───────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────┐
│   Tracking & Optimization Layer               │
│  • Plate ID tracking                          │
│  • Frame skipping                             │
│  • OCR interval control                       │
│  • Detection caching                          │
└───────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────┐
│        FastAPI Backend Server                 │
│  • REST APIs                                  │
│  • WebSocket live streaming                   │
│  • Video streaming (MJPEG)                    │
└───────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────┐
│      Frontend UI & History Storage            │
│  • Display results                            │
│  • Store logs (image/capture/video/live)      │
└───────────────────────────────────────────────┘
```
---

## 🔧 Prerequisites

```text
Software Requirements:
- Python 3.9 – 3.11
- pip (Python package manager)
- Git (for cloning repository)

Hardware Requirements:
- RAM: Minimum 8 GB recommended
- CPU: Multi-core processor
- GPU (Optional): NVIDIA GPU for faster inference

Libraries & Frameworks:
- OpenCV (cv2) → Image & video processing
- NumPy → Numerical operations
- PyTorch → Deep learning framework
- Ultralytics YOLO → Number plate detection
- NLTK → English word filtering
- FastAPI → Backend API server
- Uvicorn → ASGI server

Models Required:
- YOLO Model → best_finetuned.pt
- CRNN Model → best_crnn.pth

Other Requirements:
- Stable internet connection (for downloading models/packages)
- Webcam (for live detection, optional)
```
---

## 🚀 How to Run the Project

```text
Step 1: Clone the Repository
→ git clone <your-repo-url>
→ cd <project-folder>

Step 2: Create Virtual Environment
→ python -m venv venv

Step 3: Activate Environment
→ Windows: venv\Scripts\activate
→ Linux/Mac: source venv/bin/activate

Step 4: Install Dependencies
→ pip install -r requirements.txt

Step 5: Ensure Models are Present
→ models/best_finetuned.pt
→ models/best_crnn.pth

Step 6: Run FastAPI Server
→ uvicorn app:app --reload --host 127.0.0.1 --port 8000

Step 7: Open in Browser
→ http://localhost:8000

```
---

## 📁 Project Structure

```text

project/
│
├── app.py                     # FastAPI backend
├── speed_estimator.py         # Detection + Recognition pipeline
│
├── templates/                 # HTML UI
│   ├── index.html
│   ├── image.html
│   ├── video.html
│   ├── capture.html
│   └── live.html
│
├── static/
│   ├── uploads/               # Uploaded media
│   ├── results/               # Output images
│
├── models/
│   ├── best_finetuned.pt      # YOLO model
│   └── best_crnn.pth          # CRNN model
│
├── image_history.txt
├── video_history.txt
├── capture_history.txt
├── live_history.txt
│
└── requirements.txt
```
---

## 🖥️ Web Interface

```text
Home Page (/)
→ Overview of the system
→ Navigation to all modules

Image Detection (/image)
→ Upload an image
→ Detect number plate
→ Display extracted text and confidence
→ Save result to history

Video Processing (/video)
→ Upload video file
→ Real-time frame processing
→ Stream processed video with detections
→ Pause / Resume support
→ Save result to history

Camera Capture (/capture)
→ Capture image from browser camera
→ Perform instant detection
→ Display detected plates with confidence
→ Save result to history

Live Detection (/live)
→ Real-time detection using WebSocket
→ Continuous frame streaming
→ Optimized with frame skipping
→ Displays live plate results
→ Save result to history

```

---

## 📊 Training Results

```text
🔍 YOLOv11 (Number Plate Detection)

Precision        : 0.9708
Recall           : 0.9432
mAP@0.5          : 0.9775 
mAP@0.5:0.95     : 0.6501

Key Insights:
→ High precision indicates very few false positives
→ Strong recall ensures most plates are detected
→ High mAP@0.5 confirms robust detection performance
→ mAP@0.5:0.95 shows good performance across stricter IoU thresholds


🔤 CRNN (Text Recognition Model)

Validation Accuracy : 96.80%
Validation CER      : 0.0086

Key Insights:
→ High accuracy ensures reliable character recognition
→ Very low CER (Character Error Rate) indicates minimal OCR errors
→ Suitable for real-world number plate recognition scenarios
```

---



