import pyrealsense2 as rs
import numpy as np
import cv2

def get_dark_pink_mask(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for dark pink (adjust as needed)
    lower_dark_pink = np.array([160, 80, 80])
    upper_dark_pink = np.array([180, 255, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower_dark_pink, upper_dark_pink)
    return mask

def main():
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)

    try:
        while True:
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # Mask for dark pink
            mask = get_dark_pink_mask(color_image)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500:  # Ignore small areas
                    continue

                # Draw contour
                cv2.drawContours(color_image, [cnt], -1, (0, 255, 0), 2)

                # Get extreme points for start and end
                leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
                rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])

                cv2.circle(color_image, leftmost, 5, (255, 0, 0), -1)
                cv2.putText(color_image, "Start", (leftmost[0], leftmost[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                cv2.circle(color_image, rightmost, 5, (0, 0, 255), -1)
                cv2.putText(color_image, "End", (rightmost[0], rightmost[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Show image
            cv2.imshow('Suturing Mat Detection', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
