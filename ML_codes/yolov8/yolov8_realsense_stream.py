from flask import Flask, render_template, Response
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np

# Load YOLO model
model = YOLO("yolov8s.pt")  # Change this if you're using a different model

# Setup RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

app = Flask(__name__)

def generate_frames():
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert RealSense frame to numpy array
        frame = np.asanyarray(color_frame.get_data())

        # Run YOLOv8 inference
        results = model(frame)[0]
        annotated_frame = results.plot()

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        # Stream over HTTP
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🌐 Open your browser at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
