import cv2
import torch
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# --- Load the trained model ---
def load_model(weights_path='cuts_detector_best.pth'):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# --- RealSense setup ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# --- Stitching path functions ---
def draw_zigzag_stitch(img, box, spacing=20):
    x1, y1, x2, y2 = map(int, box)
    midline = [(int((x1 + x2) / 2), y) for y in range(y1, y2, spacing)]
    direction = 1
    for i in range(len(midline) - 1):
        pt1 = (midline[i][0] + direction * 10, midline[i][1])
        pt2 = (midline[i + 1][0] - direction * 10, midline[i + 1][1])
        cv2.line(img, pt1, pt2, (0, 0, 255), 2)
        direction *= -1
    return img

def draw_suture_arcs(img, box, spacing=20):
    x1, y1, x2, y2 = map(int, box)
    midline = [(int((x1 + x2) / 2), y) for y in range(y1, y2, spacing)]
    for pt in midline:
        center = (pt[0], pt[1])
        axes = (8, 8)
        cv2.ellipse(img, center, axes, 0, 0, 180, (0, 255, 255), 1)
    return img

# --- Flask app for browser streaming ---
app = Flask(__name__)

def detect_and_annotate(frame):
    img_tensor = F.to_tensor(frame).unsqueeze(0)
    with torch.no_grad():
        predictions = model(img_tensor)[0]

    boxes = predictions['boxes']
    scores = predictions['scores']

    for box, score in zip(boxes, scores):
        if score > 0.6:
            box = box.numpy().astype(int)
            draw_zigzag_stitch(frame, box)
            draw_suture_arcs(frame, box)

    return frame

def gen_frames():
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
        frame = detect_and_annotate(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<h1>🩺 RealSense Cut Detection</h1><img src="/video" width="640" height="480">'

if __name__ == '__main__':
    print("🌐 Visit http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=False)
