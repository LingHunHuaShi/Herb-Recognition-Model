import base64
from PIL import Image
from io import BytesIO
from predict import predict_image
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse


def inference_image_base64(image_base64):
    image = base64.b64decode(image_base64)
    img = Image.open(BytesIO(image))
    label, confidence = predict_image(img)
    return label, confidence


app = FastAPI()


@app.post("/api/predict")
async def predict(request: Request):
    body = await request.json()
    image_base64 = body.get("image")
    label, confidence = inference_image_base64(image_base64)
    return JSONResponse({"label": label, "confidence": float(confidence)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7861)
