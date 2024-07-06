import base64
from PIL import Image
import gradio as gr
from io import BytesIO
from predict import predict_image


def inference_image_base64(image_base64):
    image = base64.b64decode(image_base64)
    img = Image.open(BytesIO(image))
    label, confidence = predict_image(img)
    return label, confidence


def inference_image(image):
    label, confidence = predict_image(image)
    return label, float(confidence)


iface = gr.Interface(
    fn=inference_image,
    inputs=gr.inputs.Image(type="pil"),
    outputs=[gr.outputs.Textbox(label="Label"), gr.outputs.Textbox(label="Confidence")]
)

iface.launch(server_name="0.0.0.0", server_port=7860)
