from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
from tensorflow.keras.models import load_model
app = FastAPI()

# Load the pre-trained model
model = load_model("potatos_model1.h5")
print('model loadedd..',model)
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/")
async def ping():
    return "Hello, I am alive"

# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image

def read_file_as_image(data: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data))
        image = image.resize((256, 256))  # Resize to match the model's input size
        image = np.array(image)
        if image.shape[-1] == 4:  # Convert RGBA to RGB if necessary
            image = image[..., :3]
        image = image / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        print(f"Processed image shape: {image.shape}")  # Debugging output
        return image
    except Exception as e:
        print(f"Error in preprocessing: {e}")  # Log any preprocessing errors
        return None





@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess the image
        image = read_file_as_image(await file.read())
        
        if image is None:
            return {"error": "Image preprocessing failed."}
        
        # Make predictions
        predictions = model.predict(image)
        print(f"Predictions: {predictions}")  # Debugging output
        print(f"Image shape: {image.shape}")
        # print(f"Predictions: {predictions}")

        if predictions.size == 0:
            return {"error": "Prediction failed."}
        
        # Get the class with the highest probability
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        
        return {
            "prediction": predicted_class,
            "confidence": float(confidence)
        }
    except Exception as e:
        print(f"Error during prediction: {e}")  # Log any prediction errors
        return {"error": str(e)}


# @app.post("/predict")
# async def predict(
#     file: UploadFile = File(...)
# ):
#     image = read_file_as_image(await file.read())
#     pass
    # img_batch = np.expand_dims(image, 0)

    # json_data = {
    #     "instances": img_batch.tolist()
    # }

    # response = requests.post(endpoint, json=json_data)
    # prediction = np.array(response.json()["predictions"][0])

    # predicted_class = CLASS_NAMES[np.argmax(prediction)]
    # confidence = np.max(prediction)

    # return {
    #     "class": predicted_class,
    #     "confidence": float(confidence)
    # }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
