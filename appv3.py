import cv2
import time
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, render_template, request, jsonify, send_from_directory
from flask_sock import Sock
from threading import Thread, Lock
import socket
import asyncio
import json
import os
import base64
from datetime import datetime
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame

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
CONFIDENCE = 0.75
IOU_THRESHOLD = 0.45
TARGET_FPS = 1.0
POTHOLE_LOG_FILE = "pothole_detections.json"
SCREENSHOTS_DIR = "pothole_screenshots"

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

model = ort.InferenceSession(MODEL_FILE, providers=['CPUExecutionProvider'])

latest_frame = None
annotated_frame = None
frame_lock = Lock()
WORKER_RUNNING = True

viewer_clients_live = set()
viewer_clients_detected = set()
viewer_lock = Lock()

current_detections = {"count": 0, "max_confidence": 0.0, "boxes": []}
detection_lock = Lock()

pcs = set()
relay = MediaRelay()

loop = asyncio.new_event_loop()

def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, loop).result()

def start_background_loop():
    asyncio.set_event_loop(loop)
    loop.run_forever()

Thread(target=start_background_loop, daemon=True).start()

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

def broadcast_to_viewers(live_frame, detected_frame):
    global viewer_clients_live, viewer_clients_detected

    with viewer_lock:
        if viewer_clients_live and live_frame is not None:
            _, jpeg_live = cv2.imencode('.jpg', live_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            jpeg_bytes_live = jpeg_live.tobytes()

            dead_clients = set()
            for ws in viewer_clients_live:
                try:
                    ws.send(jpeg_bytes_live)
                except Exception:
                    dead_clients.add(ws)
            viewer_clients_live -= dead_clients

        if viewer_clients_detected and detected_frame is not None:
            _, jpeg_detected = cv2.imencode('.jpg', detected_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            jpeg_bytes_detected = jpeg_detected.tobytes()

            dead_clients = set()
            for ws in viewer_clients_detected:
                try:
                    ws.send(jpeg_bytes_detected)
                except Exception:
                    dead_clients.add(ws)
            viewer_clients_detected -= dead_clients

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

        with detection_lock:
            current_detections["count"] = len(detections)
            current_detections["max_confidence"] = max([d['confidence'] for d in detections], default=0.0)
            current_detections["boxes"] = [d['box'] for d in detections]

        broadcast_to_viewers(src, out)

        print(f"[worker] inference {(end-start)*1000:.0f} ms | detections {len(detections)}")
        elapsed = time.time() - t0
        to_sleep = period - elapsed
        if to_sleep > 0:
            time.sleep(to_sleep)

worker_thread = Thread(target=inference_worker, daemon=True)
worker_thread.start()

class VideoTransformTrack(VideoStreamTrack):

    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.counter = 0

    async def recv(self):
        global latest_frame, annotated_frame

        try:
            frame = await self.track.recv()
            img = frame.to_ndarray(format="bgr24")

            with frame_lock:
                latest_frame = img.copy()

            with frame_lock:
                out = annotated_frame.copy() if annotated_frame is not None else img.copy()

            new_frame = VideoFrame.from_ndarray(out, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base

            self.counter += 1
            if self.counter % 30 == 0:
                print(f"[VideoTransformTrack] Processed frame {self.counter}, size: {out.shape}")

            return new_frame

        except Exception as e:
            print(f"[VideoTransformTrack] Error processing frame: {e}")
            frame = await self.track.recv()
            return frame

@app.route('/')
def home():
    return render_template('indexv3.html')

@app.route('/viewer')
def viewer():
    return render_template('viewer.html')

@app.route('/map')
def pothole_map():
    return render_template('pothole_map.html')

@app.route('/screenshots/<filename>')
def serve_screenshot(filename):
    return send_from_directory(SCREENSHOTS_DIR, filename)

@sock.route('/ws_viewer_live')
def ws_viewer_live(ws):
    print("Viewer Websocket connected (live feed)")
    with viewer_lock:
        viewer_clients_live.add(ws)
    try:
        while True:
            time.sleep(1)
    except Exception as e:
        print(f"Viewer live Websocket error: {e}")
    finally:
        with viewer_lock:
            viewer_clients_live.discard(ws)
        print("Viewer Websocket disconnected (live feed)")

@sock.route('/ws_viewer_detected')
def ws_viewer_detected(ws):
    print("Viewer Websocket connected (detected feed)")
    with viewer_lock:
        viewer_clients_detected.add(ws)
    try:
        while True:
            time.sleep(1)
    except Exception as e:
        print(f"Viewer detected Websocket error: {e}")
    finally: 
        with viewer_lock:
            viewer_clients_detected.discard(ws)
        print("Viewer Websocket disconnected (detected feed)")

@app.route('/offer', methods=['POST'])
def offer():
    params = request.get_json()
    print(f"[WebRTC] Received offer from client")

    async def process_offer():
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"[WebRTC] Connection state is {pc.connectionState}")
            if pc.connectionState == "failed":
                await pc.close()
                pcs.discard(pc)

        @pc.on("track")
        def on_track(track):
            print(f"[WebRTC] Track {track.kind} received")
            if track.kind == "video":
                try:
                    relay_track = relay.subscribe(track)
                    transform_track = VideoTransformTrack(relay_track)
                    pc.addTrack(transform_track)
                    print(f"[WebRTC] Added transform track successfully")
                except Exception as e:
                    print(f"[WebRTC] Error adding track: {e}")
                    # Fallback: add original track without transformation
                    pc.addTrack(relay.subscribe(track))

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        print(f"[WebRTC] Answer created and set")

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }
    
    result = run_async(process_offer())
    return jsonify(result)

@app.route('/close', methods=['POST'])
def close():
    global latest_frame, annotated_frame

    async def close_connections():
        for pc in pcs:
            await pc.close()
        pcs.clear()

    run_async(close_connections())

    with frame_lock:
        latest_frame = None
        annotated_frame = None

    return jsonify({"status": "closed"})

@app.route('/log_pothole', methods=['POST'])
def log_pothole():
    try:
        data = request.get_json()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        screenshot_filename = f"pothole_{timestamp}.jpg"
        screenshot_path = os.path.join(SCREENSHOTS_DIR, screenshot_filename)

        if 'screenshot' in data and data['screenshot']:
            try:
                img_data = data['screenshot']
                if ',' in img_data:
                    img_data = img_data.split(',')[1]

                img_bytes = base64.b64decode(img_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                cv2.imwrite(screenshot_path, img)
                print(f"Screenshot saved: {screenshot_filename}")
            except Exception as e:
                print(f"Error saving screenshot: {e}")
                screenshot_filename = None
        else:
            screenshot_filename = None

        log_entry = {
            "id": timestamp,
            "timestamp": datetime.now().isoformat(),
            "latitude": data.get('latitude'),
            "longitude": data.get('longitude'),
            "accuracy": data.get('accuracy'),
            "confidence": data.get('confidence'),
            "detection_count": data.get('detection_count', 1),
            "bounding_boxes": data.get('bounding_boxes', []),
            "screenshot": screenshot_filename,
            "validated": None
        }

        if os.path.exists(POTHOLE_LOG_FILE):
            with open(POTHOLE_LOG_FILE, 'r') as f:
                try:
                    detections = json.load(f)
                except json.JSONDecodeError:
                    detections = []
        else:
            detections = []

        detections.append(log_entry)

        with open(POTHOLE_LOG_FILE, "w") as f:
            json.dump(detections, f, indent=2)

        print(f"[GPS] Pothole logged at ([{log_entry['latitude']}, {log_entry['longitude']}])")

        return jsonify({"status": "logged", "entry": log_entry})
    except Exception as e:
        print(f"Error logging pothole: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/validate_pothole', methods=['POST'])
def validate_pothole():
    try:
        data = request.get_json()
        pothole_id = data.get('id')
        is_valid = data.get('is_valid')

        if os.path.exists(POTHOLE_LOG_FILE):
            with open(POTHOLE_LOG_FILE, 'r') as f:
                detections = json.load(f)

            for detection in detections:
                if detection.get('id') == pothole_id:
                    detection['validated'] = is_valid
                    break
                    
            with open(POTHOLE_LOG_FILE, "w") as f:
                json.dump(detections, f, indent=2)

            return jsonify({"status": "success", "id": pothole_id, "validated": is_valid})
        else:
            return jsonify({"status": "error", "message": "Log file not found"}), 404
        
    except Exception as e:
        print(f"Error validating pothole: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/get_potholes', methods=['GET'])
def get_potholes():
    try:
        if os.path.exists(POTHOLE_LOG_FILE):
            with open(POTHOLE_LOG_FILE, 'r') as f:
                detections = json.load(f)
            return jsonify({"status": "success", "detections": detections})
        else:
            return jsonify({"status": "success", "detections": []})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/get_detection_info', methods=['GET'])
def get_detection_info():
    with detection_lock:
        return jsonify(current_detections)
    
if __name__ == '__main__':
    print("=" * 50)
    print("Starting Pothole Detection Server")
    print("=" * 50)
    print(f"Model: {MODEL_FILE}")
    print(f"Confidence Threshold: {CONFIDENCE}")
    print(f"IoU Threshold: {IOU_THRESHOLD}")
    print(f"Target push FPS: {TARGET_FPS}")
    print("=" * 50)
    print(f"Open browser on PC or phone: http://{LAN_IP}:5000")
    print(f"Desktop viewer page: http://{LAN_IP}:5000/viewer")
    print(f"Pothole map viewer:  http://{LAN_IP}:5000/map")
    print(f"GPS logs saved to:   {POTHOLE_LOG_FILE}")
    print(f"Screenshots saved to: {SCREENSHOTS_DIR}/")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    app.run(host='0.0.0.0', port=5000, debug=False)