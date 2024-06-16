from fastapi import FastAPI, UploadFile
from PIL import Image
import io

from inference import Inference
import base64

app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile):
    file_bytes = file.file.read()
    model = Inference("weights/best.pt")
    img, report = model.run(file_bytes)
    image = Image.fromarray(img)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    encoded_image = base64.b64encode(img_byte_arr).decode("utf-8")
    return {"img": encoded_image, "report": report}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8881)
