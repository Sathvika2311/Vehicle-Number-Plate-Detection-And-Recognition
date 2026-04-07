# 🚗 Vehicle Number Plate Detection & Recognition System (ANPR)

A real-time **Automatic Number Plate Recognition (ANPR)** system built using deep learning and computer vision. The system detects vehicle number plates from images, videos, and live streams, extracts text using OCR, and intelligently validates and corrects Indian license plate formats.

---

## ✨ Key Features

- 🔍 **Accurate Plate Detection**
  - YOLOv8-based number plate detection

- 🔤 **Hybrid OCR System**
  - PaddleOCR for text extraction
  - CRNN architecture support (extensible)

- 🇮🇳 **Indian Plate Validation Engine**
  - Supports Standard, BH-series, Military, Old formats
  - Regex-based validation with state verification

- 🧠 **Automatic Error Correction**
  - Fixes OCR errors (O→0, S→5, B→8, etc.)
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
  - Combines detection + OCR confidence
  - Boosting mechanism for stability

- 🧾 **History Logging**
  - Stores results with timestamp
  - Separate logs for image, video, capture, and live

---

## 🏗️ Architecture

