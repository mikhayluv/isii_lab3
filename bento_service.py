from __future__ import annotations
import gradio as gr
import typing as t
from pathlib import Path
from PIL import Image as PILImage
import os
import json
import requests
import bentoml
from bentoml.validators import ContentType

Image = t.Annotated[Path, ContentType("image/*")]
SAVE_DIR = "D:\\vscodeprojects\\ISII\\save_img"
RESULT_DIR = "D:\\vscodeprojects\\ISII\\results"

def object_detection(image: PILImage.Image):
    temp_image_path = os.path.join(SAVE_DIR, "input_image.jpg")
    print(f"temp_image_path {temp_image_path}")
    image.save(temp_image_path)
    print("IMAGE SAVED!")

    url = "http://127.0.0.1:3000/render"  # URL BentoML API
    with open(temp_image_path, "rb") as img_file:
        response = requests.post(url, files={"image": img_file})
        print(f"response: {response}")

    if response.status_code == 200:
        print(f"response {response.status_code}")
        result_image_path = os.path.join(RESULT_DIR, "input_image.jpg")
        print(f"result_image_path: {result_image_path}")
        result_img = PILImage.open(result_image_path)
        return result_img
    else:
        return None, "Ошибка: не удалось получить ответ от сервиса."
    
# Gradio UI
iface = gr.Interface(
    fn=object_detection,
    inputs=gr.Image(type="pil", label = "Upload Image"),
    outputs=gr.Image(type="pil", label = "Result"),
    title="Object Detection",
    description="Загрузите изображение здания, чтобы увидеть результат object detection.",
    css="footer{display:none !important}",
    flagging_mode="never"
)
    
@bentoml.service(resources={"gpu": 1})
class Model:
    def __init__(self):
        from ultralytics import YOLO

        self.model = YOLO("yolo11m_best.onnx")

    @bentoml.api(batchable=True)
    def predict(self, images: list[Image]) -> list[list[dict]]:
        results = self.model.predict(source=images)
        return [json.loads(result.tojson()) for result in results]
    
    @bentoml.api
    def render(self, image: Image) -> Image:
        result = self.model.predict(image)[0]
        output = image.parent.joinpath(f"{image.stem}_result{image.suffix}")
        result.save(str(output))
        return output


if __name__ == "__main__":
    iface.launch()