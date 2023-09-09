from common_imports import *
from default.industries.model_classification.libraries.prediction import Prediction #Classification Model
from default.industries.model_regression.libraries.prediction import StreamPrediction #Regression Model
from typing import List

from PIL import Image
import io
from default.vision.libraries.prediction_img import prediction_img  # Import your Prediction_IMG class
import base64

app = FastAPI()

# Create an instance of the Prediction_IMG class

Default_workspace_router = APIRouter()

@Default_workspace_router.post("/default/industries/model_c/predict")
def predict_from_ind_classification(data: dict):
    input_data = data.values()
    steam_data = [float(value) for value in input_data]
    print(steam_data)
    prediction = Prediction(steam_data=steam_data)
    make_prediction = prediction.make_prediction()
    print(make_prediction)
    return{"prediction": make_prediction}

@Default_workspace_router.post("/default/industries/model_r/predict")
def predict_from_ind_regression(data: dict):
    input_data = data.values()
    steam_data = [float(value) for value in input_data]
    # print(steam_data)
    prediction = StreamPrediction(steam_data=steam_data)
    make_prediction = prediction.make_prediction()
    print(make_prediction)
    return{"Prediction":make_prediction}


import base64
@Default_workspace_router.post("/vision/upload")
async def upload_file(file: UploadFile):
    try:
        # Set a chunk size (e.g., 4096 bytes)
        chunk_size = 4096
        image_bytes = b''

        # Read and process the image in chunks
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            image_bytes += chunk

        # Create an Image object from image_bytes
        image = Image.open(io.BytesIO(image_bytes))

        # Encode the uploaded image as base64
        image_base64 = base64.b64encode(image_bytes).decode()

        # Make predictions using your Prediction_IMG class
        predictions = prediction_img.make_prediction(image)

        # Save the image
        prediction_image_path = prediction_img.get_latest_prediction_img()

        with open(prediction_image_path, "rb") as image_file:
            prediction_image_bytes = image_file.read()

        prediction_image_base64 = base64.b64encode(prediction_image_bytes).decode()

        response_data = {
            "predictions": predictions,
            "uploaded_image_base64": image_base64,
            "prediction_image_base64": prediction_image_base64
        }

        return JSONResponse(content=response_data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
