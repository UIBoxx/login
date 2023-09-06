from common_imports import *
from default.industries.model1.libraries.prediction import Prediction

Default_workspace_router = APIRouter()

@Default_workspace_router.post("/default/industries/model1/predict")
def predict_from_ind_classification(data: dict):
    input_data = data.values()
    steam_data = [float(value) for value in input_data]
    print(steam_data)
    prediction = Prediction(steam_data=steam_data)
    make_prediction = prediction.make_prediction()
    print(make_prediction)
    return{"prediction": make_prediction}
