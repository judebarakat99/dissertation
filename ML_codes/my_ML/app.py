import cv2
import torch
import numpy as np
from flask import Flask, Response, render_template_string
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pyrealsense2 as rs

# Flask setup
app = Flask(__name__)

# Load model
def load_model(weights_path="cuts_detector_best.pth"):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

model = load_model()

# RealSense camera setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# HTML Template
html_template = """
<!DOCTYPE html>
<html>
<head><title>Cut Detection & Stitching</title></head>
<body style="display: flex; gap: 20px;">
    <div>
        <h2>Wound Detection (Mask)</h2>
        <img src="{{ url_for('video_feed_mask') }}" width="640" />
    </div>
    <div>
        <h2>Stitch Path</h2>
        <img src="{{ url_for('video_feed_stitch') }}" width="640" />
    </div>
</body>
</html>
"""

# Utility: Run model and get boxes
def detect_cuts(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)[0]
    boxes = output['boxes']
    scores = output['scores']
    return [box for box, score in zip(boxes, scores) if score > 0.5]

# Draw mask (rectangle)
def draw_mask(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

# Zig-zag stitch pattern
def draw_stitch(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = box.int().tolist()
        mid_x = (x1 + x2) // 2
        spacing = 20
        points = []

        for y in range(y1, y2, spacing):
            offset = 10 if (y // spacing) % 2 == 0 else -10
            points.append((mid_x + offset, y))
            cv2.circle(frame, (mid_x + offset, y), 4, (0, 0, 255), -1)

        # Draw connecting lines
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], (255, 0, 0), 2)

    return frame

# Video generator for masked detection
def gen_mask():
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        boxes = detect_cuts(frame.copy())
        frame = draw_mask(frame, boxes)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Video generator for stitching
def gen_stitch():
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        boxes = detect_cuts(frame.copy())
        frame = draw_stitch(frame, boxes)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Flask Routes
@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/video_feed_mask')
def video_feed_mask():
    return Response(gen_mask(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_stitch')
def video_feed_stitch():
    return Response(gen_stitch(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start Flask app
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        pipeline.stop()
