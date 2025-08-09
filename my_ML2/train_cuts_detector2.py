import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from flask import Flask, Response, render_template_string, request, jsonify
import time

# =============================
# CONFIGURATION & SETTINGS
# =============================
CONFIG = {
    "process_every_n": 3,  # Only process every Nth frame
    "conf_threshold": 0.5,
    "overlay_mode": "both",  # "mask", "stitch", "both"
}

# =============================
# MODEL LOADING (Mask R-CNN)
# =============================
model = torch.load("mask_cuts_detector.pth", map_location="cpu")
if isinstance(model, torch.nn.Module):
    pass
else:
    raise ValueError("Loaded object is not a PyTorch model")
model.eval()

# =============================
# REALSENSE CAMERA INIT
# =============================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# =============================
# FLASK SERVER
# =============================
app = Flask(__name__)

# =============================
# FPS TRACKING
# =============================
frame_count = 0
last_time = time.time()
fps = 0.0

# =============================
# FRAME GENERATOR FUNCTION
# =============================
def generate_frames():
    global frame_count, last_time, fps
    skip_counter = 0
    last_result_mask = None
    last_result_stitch = None

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        skip_counter += 1

        # FPS calculation
        frame_count += 1
        if frame_count >= 10:
            now = time.time()
            fps = frame_count / (now - last_time)
            last_time = now
            frame_count = 0

        if skip_counter % CONFIG["process_every_n"] == 0:
            # Convert to tensor for model
            img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)[0]

            masks = outputs['masks'] > CONFIG["conf_threshold"]
            stitched_img = frame.copy()
            mask_img = frame.copy()

            for mask in masks:
                mask_np = mask[0].cpu().numpy().astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(mask_img, contours, -1, (0, 255, 0), 2)

                # Draw single continuous stitch pattern with circles
                for contour in contours:
                    pts = contour[::20]  # sample points for stitches
                    for p in pts:
                        cv2.circle(stitched_img, tuple(p[0]), 3, (0, 0, 255), -1)

            last_result_mask = mask_img
            last_result_stitch = stitched_img
        
        # Choose overlay
        if CONFIG["overlay_mode"] == "mask":
            display_frame = last_result_mask if last_result_mask is not None else frame
        elif CONFIG["overlay_mode"] == "stitch":
            display_frame = last_result_stitch if last_result_stitch is not None else frame
        else:  # both
            combined = np.hstack((last_result_mask, last_result_stitch)) if last_result_mask is not None else frame
            display_frame = combined

        # Draw FPS
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', display_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# =============================
# ROUTES
# =============================
@app.route('/')
def index():
    return render_template_string('''
    <html>
    <head>
        <title>RealSense Stitch Detection</title>
        <script>
            function updateSetting(name, value) {
                fetch(`/update?name=${name}&value=${value}`)
            }
        </script>
    </head>
    <body>
        <h1>RealSense Stitch Detection</h1>
        <img src="/video_feed" width="1280">
        <div>
            <label>Overlay Mode:</label>
            <select onchange="updateSetting('overlay_mode', this.value)">
                <option value="mask">Mask Only</option>
                <option value="stitch">Stitch Only</option>
                <option value="both" selected>Both</option>
            </select>
            <br>
            <label>Confidence Threshold:</label>
            <input type="range" min="0" max="1" step="0.05" value="0.5" onchange="updateSetting('conf_threshold', this.value)">
            <br>
            <label>Process Every N Frames:</label>
            <input type="number" value="3" min="1" max="10" onchange="updateSetting('process_every_n', this.value)">
        </div>
    </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update')
def update():
    name = request.args.get('name')
    value = request.args.get('value')
    if name in CONFIG:
        if name in ["process_every_n"]:
            CONFIG[name] = int(value)
        elif name in ["conf_threshold"]:
            CONFIG[name] = float(value)
        else:
            CONFIG[name] = value
    return jsonify(CONFIG)

# =============================
# MAIN ENTRY
# =============================
if __name__ == '__main__':
    print("Starting server at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
