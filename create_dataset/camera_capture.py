import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

# === Configuration ===
total_images = 200
delay_seconds = 1
rgb_folder = "raw_images_RGB2"
depth_folder = "raw_images_depth2"

# Create folders if they don't exist
os.makedirs(rgb_folder, exist_ok=True)
os.makedirs(depth_folder, exist_ok=True)

# === Initialize RealSense pipeline ===
pipeline = rs.pipeline()
config = rs.config()

# Enable RGB and depth streams
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)
time.sleep(2.0)  # Let camera stabilize

print("[INFO] Starting capture and visualization...")

try:
    start_time = time.time()

    for i in range(total_images):
        # Get frameset
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print("[WARN] Skipping frame...")
            continue

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert depth to color map for visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Stack RGB and depth side-by-side
        images_combined = np.hstack((color_image, depth_colormap))

        # Show image in window
        cv2.imshow("RGB (Left) | Depth (Right)", images_combined)

        # Handle exit key
        key = cv2.waitKey(1)
        if key == 27:  # ESC key to quit early
            print("[INFO] Interrupted by user.")
            break

        # Filename with timestamp
        elapsed = time.time() - start_time
        seconds = int(elapsed)
        tenths = int((elapsed - seconds) * 10)
        filename = f"img-{seconds}.{tenths}.png"

        # Save both images
        cv2.imwrite(os.path.join(rgb_folder, filename), color_image)
        cv2.imwrite(os.path.join(depth_folder, filename), depth_colormap)

        print(f"[INFO] Saved {filename}")
        time.sleep(delay_seconds)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] Finished capturing and displaying.")

