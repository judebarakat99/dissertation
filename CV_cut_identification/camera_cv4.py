import cv2
import numpy as np
import pyrealsense2 as rs

def nothing(x):
    pass

def create_hsv_trackbars():
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("H min", "Trackbars", 0, 180, nothing)
    cv2.createTrackbar("H max", "Trackbars", 10, 180, nothing)
    cv2.createTrackbar("S min", "Trackbars", 100, 255, nothing)
    cv2.createTrackbar("S max", "Trackbars", 180, 255, nothing)
    cv2.createTrackbar("V min", "Trackbars", 80, 255, nothing)
    cv2.createTrackbar("V max", "Trackbars", 170, 255, nothing)

def get_hsv_bounds():
    h_min = cv2.getTrackbarPos("H min", "Trackbars")
    h_max = cv2.getTrackbarPos("H max", "Trackbars")
    s_min = cv2.getTrackbarPos("S min", "Trackbars")
    s_max = cv2.getTrackbarPos("S max", "Trackbars")
    v_min = cv2.getTrackbarPos("V min", "Trackbars")
    v_max = cv2.getTrackbarPos("V max", "Trackbars")
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    return lower, upper

def main():
    # Initialize RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale:", depth_scale)

    create_hsv_trackbars()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            # Convert BGR to HSV
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            lower, upper = get_hsv_bounds()

            # Threshold wound mask
            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(color_image, color_image, mask=mask)

            #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # --- Enhance the mask before finding contours ---
            # Fill small holes
            kernel = np.ones((5, 5), np.uint8)
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Smooth jagged edges
            mask_blur = cv2.GaussianBlur(mask_clean, (5, 5), 0)

            # Re-binarize blurred mask
            _, mask_thresh = cv2.threshold(mask_blur, 50, 255, cv2.THRESH_BINARY)

            # Optional: slightly expand the mask to capture full boundary
            mask_dilated = cv2.dilate(mask_thresh, kernel, iterations=1)

            # Find contours on cleaned-up mask
            contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500:
                    continue

                # Draw contour
                cv2.drawContours(color_image, [cnt], -1, (0, 255, 0), 2)

                leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
                rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])

                # Draw keypoints
                cv2.circle(color_image, leftmost, 5, (255, 0, 0), -1)
                cv2.circle(color_image, rightmost, 5, (0, 0, 255), -1)
                cv2.putText(color_image, "Start", (leftmost[0], leftmost[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(color_image, "End", (rightmost[0], rightmost[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Get depth and 3D coordinates
                z1 = depth_frame.get_distance(leftmost[0], leftmost[1])
                z2 = depth_frame.get_distance(rightmost[0], rightmost[1])

                p1_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [leftmost[0], leftmost[1]], z1)
                p2_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [rightmost[0], rightmost[1]], z2)

                # Log coordinates
                print(f"Start Point (3D): {p1_3d}")
                print(f"End Point   (3D): {p2_3d}")

            # Show windows
            cv2.imshow("RGB with Wound Overlay", color_image)
            cv2.imshow("Wound Mask", mask)
            cv2.imshow("Filtered Region", result)
            cv2.imshow("Cleaned Mask", mask_dilated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
