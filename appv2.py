import cv2
import time
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, render_template
from waitress import serve
from threading import Thread, Lock
from flask_sock import Sock
import socket

app= Flask(__name__)
sock = Sock(app)

def get_lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

LAN_IP = get_lan_ip()
print(f"Server LAN IP: {LAN_IP}")

MODEL_FILE = "best.onnx"
CONFIDENCE = 0.65
IOU_THRESHOLD = 0.45
TARGET_FPS = 4.0

model = ort.InferenceSession(MODEL_FILE, providers=['CPUExecutionProvider'])

latest_frame = None
annotated_frame = None
frame_lock = Lock()
WORKER_RUNNING = True

def prepare_image(frame):
    h, w = frame.shape[:2]

    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) /255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    return img, w, h

def non_max_suppression(detections, iou_threshold=IOU_THRESHOLD):
    if len(detections) == 0:
        return []
    
    boxes = []
    scores = []
    
    for det in detections:
        x1, y1, x2, y2 = det['box']
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(det['confidence'])

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold=0.0,
        nms_threshold=iou_threshold
    )

    if len(indices) == 0:
        return []
    
    if isinstance(indices, tuple):
        indices = indices[0]

    kept_detections = []

    for i in indices:
        idx = i[0] if isinstance(i,(list, np.ndarray)) else int(i)
        kept_detections.append(detections[idx])

    return kept_detections

def detect_potholes(frame):
    input_image, orig_w, orig_h = prepare_image(frame)

    output = model.run(None, {"images": input_image})[0]

    detections = []
    output = output[0].T

    for row in output:
        
        confidence = row[4:].max()
        if confidence < CONFIDENCE:
            continue

        xc, yc, w, h = row[:4]

        x1 = int((xc - w/2) * orig_w /640)
        y1 = int((yc - h/2) * orig_h /640)
        x2 = int((xc + w/2) * orig_w /640)
        y2 = int((yc + h/2) * orig_h /640)

        detections.append({
            'box': (x1, y1, x2, y2),
            'confidence': float(confidence)
        })
    
    detections = non_max_suppression(detections, IOU_THRESHOLD)
    return detections

def draw_boxes(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det['box']
        conf = det['confidence']

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"Pothole {conf:.2f}"
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0 ), 3)
        
    return frame

def inference_worker():
    global latest_frame, annotated_frame, WORKER_RUNNING
    period = 1.0 / TARGET_FPS
    print(f"Inference worker started (target {TARGET_FPS} FPS, period {period:.3f}s)")
    while WORKER_RUNNING:
        t0 = time.time()
        with frame_lock:
            src = None if latest_frame is None else latest_frame.copy()
        if src is None:
            time.sleep(0.01)
            continue

        start = time.time()
        detections = detect_potholes(src)
        end = time.time()
        out = draw_boxes(src.copy(), detections)
        with frame_lock:
            annotated_frame = out
        print(f"[worker] inference {(end-start)*1000:.0f} ms | detections {len(detections)}")
        elapsed = time.time() - t0
        to_sleep = period - elapsed
        if to_sleep > 0:
            time.sleep(to_sleep)

worker_thread = Thread(target=inference_worker, daemon=True)
worker_thread.start()

@sock.route('/ws')
def ws_frame_receiver(ws):
    global latest_frame
    print("Websocket connected! (receiver)")

    try:
        while True:
            frame_bytes = ws.receive()
            if frame_bytes is None:
                print("Websocket (receiver) disconnected.")
                break
            
            if isinstance(frame_bytes, str):
                continue

            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            with frame_lock:
                latest_frame = frame
    except Exception as e:
        print("ws_frame_receiver error:", e)

@sock.route('/ws_out')
def ws_frame_sender(ws):
    global latest_frame
    print("Websocket connected! (sender)")
    try:
        while True:
            with frame_lock:
                out = None if annotated_frame is None else annotated_frame.copy()
            if out is None:
                time.sleep(0.02)
                continue

            _, jpeg = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            try:
                ws.send(jpeg.tobytes())
            except Exception:
                print("ws_out: client disconnected or send error")
                break
            time.sleep(0.03)
    except Exception as e:
        print("ws_frame_sender error:", e)

@app.route('/')
def home():
    return render_template('indexv2.html')

if __name__ == '__main__':
    print("=" * 50)
    print("Starting Pothole Detection Server")
    print("=" * 50)
    print(f"Model: {MODEL_FILE}")
    print(f"Confidence Threshold: {CONFIDENCE}")
    print(f"IoU Threshold: {IOU_THRESHOLD}")
    print(f"Target push FPS (ws_out): {TARGET_FPS}")
    print("=" * 50)
    print(f"Open browser on PC or phone: http://{LAN_IP}:5000")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    app.run(host='0.0.0.0', port=5000, debug=True)