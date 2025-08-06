from ultralytics import YOLO

# Load a YOLOv8n model (nano version, fastest for CPU)
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='create_dataset/raw_images_RGB_yolov8/data.yaml',  # relative to this script
    epochs=50,
    imgsz=640,
    batch=8,
    name='wound_detector',
    project='runs/detect',
    verbose=True
)

# Optional: Evaluate model performance on validation set
metrics = model.val()
