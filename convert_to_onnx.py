from ultralytics import YOLO

model = YOLO("yolo11m_best.pt")
model.export(format="onnx")  