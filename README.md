🚗 Vehicle Number Plate Detection and Recognition System Using Deep Learning

A real-time Automatic Number Plate Detection and Recognition system built using deep learning and computer vision. The system detects vehicle number plates from images, videos, and live streams using YOLOv11, extracts text using CRNN, and intelligently validates and corrects Indian license plate formats.
✨ Key Features
🔍 Accurate Plate Detection
Uses YOLOv11 for fast and reliable number plate localization
🔤 Hybrid Text Recognition System
CRNN for robust text extraction
🇮🇳 Indian Plate Validation Engine
Supports Standard, BH-series, Military, and Old formats
Regex-based validation with state code verification
🧠 Automatic Error Correction
Fixes OCR mistakes (e.g., O→0, S→5, B→8)
Format-aware correction logic
🎥 Multi-Input Support
Image upload
Video file processing
Webcam capture
Live streaming via WebSocket
⚡ Real-Time Optimization
Frame skipping for faster processing
OCR interval control
Detection caching across frames
📊 Confidence Scoring
Combines detection confidence + OCR confidence
Boosting mechanism for stability
🧾 History Logging
Stores detected plates with timestamp
Separate logs for image, video, capture, and live modes

🏗️ Architecture

The system is divided into multiple layers for efficient processing:
Input Sources
(Image / Video / Live Stream)
            │
            ▼
YOLOv11 Detection Model
(Detects number plate regions)
            │
            ▼
OCR Processing Layer
(CRNN)
            │
            ▼
Text Filtering & Cleaning
(Remove noise, symbols, invalid words)
            │
            ▼
Indian Plate Validation Engine
(Regex + format-specific correction)
            │
            ▼
Tracking & Frame Optimization
(ID matching, OCR interval, caching)
            │
            ▼
FastAPI Backend
(API + WebSocket Streaming)
            │
            ▼
Frontend UI + History Storage

⚙️ Installation
