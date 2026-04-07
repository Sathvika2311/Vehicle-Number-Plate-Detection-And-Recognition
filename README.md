# 🚗 Vehicle Number Plate Detection and Recognition System Using Deep Learning

A real-time **Automatic Number Plate Detection and Recognition** system built using deep learning and computer vision. The system detects vehicle number plates from images, videos, and live streams, extracts text using OCR, and intelligently validates and corrects Indian license plate formats.

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

## ⚙️ Installation

```text

# 1. Clone the repository
git clone <your-repo-url>
cd <project-folder>

# 2. Create virtual environment
python -m venv venv

# Activate environment

# Windows:
venv\Scripts\activate

# Linux / Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
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
