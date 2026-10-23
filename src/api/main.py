from fastapi import FastAPI, File, UploadFile
from src.api.predictor import ImagePredictor
from PIL import UnidentifiedImageError


app = FastAPI(title = "ButterflyClassifier")

predictor_config_path = "config.yml"
predictor = ImagePredictor.init_from_config_path(predictor_config_path)

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    return predictor.predict_from_file(file.file)
