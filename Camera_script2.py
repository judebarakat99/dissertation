import cv2
from ultralytics import YOLO

# Load your trained YOLOv8 model (replace path if needed)
model = YOLO("runs/detect/wound_detector/weights/best.pt")

# Start video capture (0 = default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("✅ Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run detection
    results = model.predict(source=frame, conf=0.3, show=False, stream=True)

    for r in results:
        annotated_frame = r.plot()  # draws boxes, labels, etc.

        # Display the frame
        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()
