from ultralytics import YOLO
import matplotlib.pyplot as plt

# === Train the model ===
model = YOLO("yolov8n.yaml")  # YOLOv8 nano model (fast + small)

# Dataset config
# Make sure raw_images_RGB_annotated_yolov8/data.yaml exists and is correct
results = model.train(
    data="raw_images_RGB_annotated_yolov8/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="wound_detector",
)

# === Save training accuracy/loss graph ===
metrics = results.results_dict

plt.figure(figsize=(10, 6))
plt.plot(metrics["metrics/precision(B)"], label="Precision")
plt.plot(metrics["metrics/recall(B)"], label="Recall")
plt.plot(metrics["metrics/mAP50(B)"], label="mAP@0.5")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("Training Metrics")
plt.legend()
plt.grid()
plt.savefig("wound_training_metrics.png")
plt.show()
