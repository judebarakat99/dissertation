import pyrealsense2 as rs
import numpy as np
import cv2

def get_wound_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Refined HSV range based on your RGB data
    lower_wound = np.array([0, 100, 80])   # H, S, V
    upper_wound = np.array([10, 180, 170])

    mask = cv2.inRange(hsv, lower_wound, upper_wound)
    return mask

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            mask = get_wound_mask(color_image)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500:
                    continue

                # Draw contour
                cv2.drawContours(color_image, [cnt], -1, (0, 255, 0), 2)

                leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
                rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])

                cv2.circle(color_image, leftmost, 5, (255, 0, 0), -1)
                cv2.putText(color_image, "Start", (leftmost[0], leftmost[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                cv2.circle(color_image, rightmost, 5, (0, 0, 255), -1)
                cv2.putText(color_image, "End", (rightmost[0], rightmost[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Display result
            cv2.imshow("Wound Detection", color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
