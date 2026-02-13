import os
import uuid
import cv2
import base64
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from speed_estimator import speed_obj as speed_obj_global, SpeedEstimator


app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static/results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/image", response_class=HTMLResponse)
async def image_page(request: Request):
    return templates.TemplateResponse("image.html", {"request": request})

@app.post("/image")
async def image(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    img_bytes = await image.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Invalid image file"}, status_code=400)

    speed_obj = SpeedEstimator(speed_obj_global.model, speed_obj_global.region)
    result_bytes = speed_obj.detect_frame(img, force_ocr=True, max_dim=1024)

    if result_bytes is None:
        return {"no_plate": True}

    result_name = f"{uuid.uuid4().hex}.jpg"
    with open(os.path.join(RESULT_FOLDER, result_name), "wb") as f:
        f.write(result_bytes)

    return {"no_plate": False, "result": result_name}


@app.get("/video", response_class=HTMLResponse)
async def video_page(request: Request):
    return templates.TemplateResponse("video.html", {"request": request})

@app.post("/video")
async def video(video: UploadFile = File(...)):
    if not video.filename:
        return JSONResponse({"error": "No video selected"}, status_code=400)

    if video.filename.split(".")[-1].lower() not in {"mp4", "avi", "mov", "mkv"}:
        return JSONResponse({"error": "Only video files allowed"}, status_code=400)

    if not video.content_type.startswith("video/"):
        return JSONResponse({"error": "Invalid video type"}, status_code=400)

    video_name = f"{uuid.uuid4().hex}.mp4"
    path = os.path.join(UPLOAD_FOLDER, video_name)

    data = await video.read()
    with open(path, "wb") as f:
        f.write(data)

    return {"video_path": video_name}


@app.get("/video_feed/{video_file}")
def video_feed(video_file: str):
    path = os.path.join(UPLOAD_FOLDER, video_file)
    if not os.path.exists(path):
        return Response("Video not found", status_code=404)

    return StreamingResponse(
        generate_video_frames(path),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/video_pause/{video_file}")
def video_pause(video_file: str):
    path = os.path.join(UPLOAD_FOLDER, video_file)
    state = video_states.get(path)
    if state:
        state["paused"] = True
    return {"status": "paused"}


@app.post("/video_resume/{video_file}")
def video_resume(video_file: str):
    path = os.path.join(UPLOAD_FOLDER, video_file)
    state = video_states.get(path)
    if state:
        state["paused"] = False
    return {"status": "resumed"}


VIDEO_FRAME_SKIP = 3
video_states = {}

def generate_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    video_states[video_path] = {"paused": False}
    speed_obj = SpeedEstimator(speed_obj_global.model, speed_obj_global.region)

    frame_id = 0
    last_processed = None

    while cap.isOpened():

        state = video_states.get(video_path)

        if state and state["paused"]:
            if last_processed is not None:
                ret2, buffer = cv2.imencode(".jpg", last_processed)
                if ret2:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + buffer.tobytes() +
                        b"\r\n"
                    )
            continue

        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        if frame_id % VIDEO_FRAME_SKIP != 0 and last_processed is not None:
            output = last_processed
        else:
            output, _ = speed_obj.detect_frame_np(frame, force_ocr=False)
            last_processed = output

        ret2, buffer = cv2.imencode(".jpg", output)
        if not ret2:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes() +
            b"\r\n"
        )

    cap.release()
    video_states.pop(video_path, None)




@app.get("/capture", response_class=HTMLResponse)
async def capture_page(request: Request):
    return templates.TemplateResponse(
        "capture.html",
        {"request": request, "result": None, "no_plate": False}
    )

@app.post("/capture", response_class=HTMLResponse)
async def capture(request: Request, image: str = Form(...)):
    header, encoded = image.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    speed_obj = SpeedEstimator(speed_obj_global.model, speed_obj_global.region)
    result_bytes = speed_obj.detect_frame(img, force_ocr=True, max_dim=1024)

    no_plate = False
    result_name = None

    if result_bytes is None:
        no_plate = True
    else:
        result_name = f"{uuid.uuid4().hex}.jpg"
        with open(os.path.join(RESULT_FOLDER, result_name), "wb") as f:
            f.write(result_bytes)

    return templates.TemplateResponse(
        "capture.html",
        {"request": request, "result": result_name, "no_plate": no_plate}
    )


@app.get("/live", response_class=HTMLResponse)
async def live(request: Request):
    return templates.TemplateResponse("live.html", {"request": request})


LIVE_FRAME_SKIP = 3   
@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()

    speed_obj = SpeedEstimator(
        speed_obj_global.model,
        speed_obj_global.region
    )

    processing = False  

    try:
        while True:
            data = await websocket.receive_text()

            if data == "__STOP__":
                break

            
            if processing:
                continue

            processing = True

            header, encoded = data.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                processing = False
                continue

            try:
                processed, _ = speed_obj.detect_frame_np(
                    frame,
                    force_ocr=False
                )
            except:
                processed = frame

            
            h, w = processed.shape[:2]
            if w > 640:
                scale = 640 / w
                processed = cv2.resize(processed, (640, int(h * scale)))

            ok, buffer = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                await websocket.send_bytes(buffer.tobytes())

            processing = False   

    except WebSocketDisconnect:
        pass

