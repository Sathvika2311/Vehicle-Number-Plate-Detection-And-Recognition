from ultralytics import YOLO

model = YOLO("best.pt")

print("Class names:", model.names)
