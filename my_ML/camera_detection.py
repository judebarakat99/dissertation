import cv2
import torch
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from flask import Flask, Response, render_template_string

# Try importing RealSense
try:
    import pyrealsense2 as rs
    realsense_available = True
except ImportError:
    realsense_available = False

# --- Load model ---
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def preprocess(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return F.to_tensor(rgb)

def draw_boxes(frame, boxes, scores, threshold=0.5):
    for box, score in zip(boxes, scores):
        if score > threshold:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Cut: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# --- Flask setup ---
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=2)
model.load_state_dict(torch.load("cuts_detector_best.pth", map_location=device))
model.to(device)
model.eval()

# --- Initialize camera ---
if realsense_available:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
else:
    cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        if realsense_available:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
        else:
            ret, frame = cap.read()
            if not ret:
                break

        # Inference
        img_tensor = preprocess(frame).to(device)
        with torch.no_grad():
            prediction = model([img_tensor])[0]

        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        frame = draw_boxes(frame, boxes, scores)

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string('''
        <html>
        <head><title>Live Cut Detection</title></head>
        <body>
            <h1>Live Cut Detection Stream</h1>
            <img src="{{ url_for('video_feed') }}" width="640" height="480">
        </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Start server ---
if __name__ == '__main__':
    print("🌐 Visit http://localhost:5000 to see the live feed")
    app.run(host='0.0.0.0', port=5000, debug=False)
