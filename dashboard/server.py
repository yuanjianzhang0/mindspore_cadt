import warnings
warnings.filterwarnings('ignore')

import os
import json
import time
import random
import asyncio
from fastapi import FastAPI, WebSocket, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="CADT Coastal Command Center")

# 绝对路径配置，防止找不到模板
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

clients = set()

# 全局状态跟踪
last_real_data_time = 0.0
simulated_prob = 0.1

async def simulation_engine():
    """
    后台驻留的自动模拟引擎。
    如果超过 2 秒没有收到真实的 NPU 推理数据，则自动生成逼真的模拟数据，保持大屏活力。
    """
    global simulated_prob
    while True:
        await asyncio.sleep(0.1)  # 10Hz 刷新率
        
        # 检查是否丢失真实心跳
        if time.time() - last_real_data_time > 2.0:
            # --- 逼真的模拟数据生成逻辑 ---
            # 概率随机游走 (平滑波动)
            simulated_prob = max(0.01, min(0.99, simulated_prob + random.uniform(-0.02, 0.02)))
            
            # 2% 的概率模拟一次突发性溺水险情
            if random.random() < 0.02:
                simulated_prob = random.uniform(0.85, 0.98)
            
            # 随着溺水概率上升，心率波动率(HRV)骤降
            hrv_baseline = 0.8 if simulated_prob < 0.5 else 0.3
            simulated_hrv = max(0.1, hrv_baseline + random.uniform(-0.1, 0.1))
            
            data = {
                "mode": "SIMULATION (自动模拟)",
                "distress_prob": simulated_prob,
                "physio_hrv": simulated_hrv,
                "latency_ms": random.uniform(25.0, 38.0),
                "qwen_strategy": None,
                "swarm_status": None
            }
            
            # 触发模拟的大模型救援指令
            if simulated_prob >= 0.85:
                data["qwen_strategy"] = {
                    "severity_level": "CRITICAL", 
                    "primary_action": "DEPLOY_AED_PAYLOAD",
                    "reasoning": "Detected erratic HRV coupled with continuous sub-surface struggling."
                }
                data["swarm_status"] = f"[USV-1] 高速前往坐标: 34.21°N, 118.{random.randint(10,99)}°E\n[UUV-2] 已下潜确认水下身姿"

            # 广播给前端
            if clients:
                message = json.dumps(data)
                disconnected = set()
                for client in clients:
                    try:
                        await client.send_text(message)
                    except:
                        disconnected.add(client)
                clients.difference_update(disconnected)

@app.on_event("startup")
async def startup_event():
    # 启动应用时，拉起模拟引擎后台任务
    asyncio.create_task(simulation_engine())

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
    """接收来自真实 NPU 和大模型的推送"""
    global last_real_data_time
    last_real_data_time = time.time() # 刷新真实数据心跳
    
    data["mode"] = "REAL-TIME (实机接入)"
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
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")