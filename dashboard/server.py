import warnings
warnings.filterwarnings('ignore')

import os
import cv2
import json
from fastapi import FastAPI, WebSocket, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

app = FastAPI(title="CADT Coastal Command Center - STRICT REAL-TIME MODE")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

clients = set()

def generate_video_frames(camera_url: str):
    cap = cv2.VideoCapture(camera_url)
    
    if not cap.isOpened():
        print(f"[ERROR] 无法连接到真实视频源: {camera_url}")
        return

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/api/video_feed")
async def video_feed(url: str):
    print(f"[INFO] 正在建立真实视频流传输通道: {url}")
    return StreamingResponse(
        generate_video_frames(url), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        clients.remove(websocket)

@app.post("/api/push_telemetry")
async def push_telemetry(data: dict):

    message = json.dumps(data)
    disconnected = set()
    for client in clients:
        try:
            await client.send_text(message)
        except:
            disconnected.add(client)
    clients.difference_update(disconnected)
    return {"status": "success"}

if __name__ == "__main__":
    print("[INFO] 工业指挥大屏后端已启动 (纯实机模式) -> http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")