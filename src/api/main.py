from fastapi import FastAPI, File, UploadFile, HTTPException
from src.api.predictor import ImageClassifier
from PIL import UnidentifiedImageError


app = FastAPI(title="Butterfly Classifier")

predictor_config_path = "config.yml"
predictor = ImageClassifier.init_from_config_path(predictor_config_path)

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        return {"predictions": predictor.predict_from_file(file.file)}
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read the image.")

@app.get("/status")
def status():
    return {"status": "working"}