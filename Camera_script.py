import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO

# === Load the trained YOLOv8 model ===
model = YOLO("runs/detect/wound_detector/weights/best.pt")

# === Initialize RealSense camera ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start camera stream
pipeline.start(config)

print("[INFO] RealSense camera started. Press ESC to exit.")

try:
    while True:
        # Get frames from the camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert RealSense frame to numpy array
        frame = np.asanyarray(color_frame.get_data())

        # Run YOLO inference
        results = model.predict(frame, imgsz=640, conf=0.3, verbose=False)
        annotated_frame = results[0].plot()  # Draw boxes and labels

        # Show the result
        cv2.imshow("Wound Detection - Live", annotated_frame)

        # Exit on ESC key
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] Camera stopped and window closed.")
