import gradio as gr
from PIL import Image
import os
from ultralytics import YOLO

model = YOLO("yolo11m_best.onnx")

# Директория для сохранения результатов
SAVE_DIR = "D:\\vscodeprojects\\ISII\\save_img"

def object_detection(image: Image.Image, conf_threshold):
    # Сохраняем входное изображение во временный файл
    temp_image_path = os.path.join(SAVE_DIR, "input_image.jpg")
    image.save(temp_image_path)

    # Запускаем предсказание
    result = model.predict(source=temp_image_path, 
                           project=SAVE_DIR, 
                           name=SAVE_DIR, 
                           exist_ok=True,  
                           save=True, 
                           save_conf=True, 
                           conf=conf_threshold, 
                           line_width=1)

    result_image_path = os.path.join(SAVE_DIR, "input_image.jpg")
    annotated_image = Image.open(result_image_path)

    # Считаем статистику по классам
    stats = [r.verbose() for r in result]
    result = ', '.join(stats)

    return annotated_image, result[:-2]

# Gradio UI
iface = gr.Interface(
    fn=object_detection,
    inputs=[
        gr.Image(type="pil", label = "Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
    ],
    outputs=[
        gr.Image(type="pil", label = "Result"), 
        gr.Textbox(label = "Classes")
    ],
    title="Object Detection",
    description="Загрузите изображение здания, чтобы увидеть результат object detection.",
    css="footer{display:none !important}",
    flagging_mode="never"
)

if __name__ == "__main__":
    iface.launch()