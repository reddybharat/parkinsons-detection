import os
import shutil
import json
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from src.api.models import ModelsResponse, PredictionResponse
from src.predictions.prediction import Predict
from PIL import Image  # Added for image validation

# Setup logging (if not imported from main)
logger = logging.getLogger("parkinsons_api")

# Config (should match main.py)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models'))
METRICS_PATH = os.path.join(MODELS_DIR, 'metrics.json')
TEMP_UPLOAD_DIR = os.getenv("TEMP_UPLOAD_DIR", "temp_uploads")

# Read available models from metrics.json
try:
    with open(METRICS_PATH, 'r') as f:
        METRICS = json.load(f)
    AVAILABLE_MODELS = list(METRICS.keys())
except Exception as e:
    logger.error(f"Failed to load metrics: {e}")
    METRICS = {}
    AVAILABLE_MODELS = []

router = APIRouter()

@router.get("/models", tags=["Models"])
def get_models():
    """List available models and their metrics."""
    if not METRICS:
        logger.error("No metrics found.")
        raise HTTPException(status_code=500, detail="Model metrics not available.")
    return {"models": METRICS}

@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(
    file: UploadFile = File(..., description="Image file (PNG/JPG)"),
    model_name: str = Form(..., description="Model display name, e.g. Linear SVM")
):
    """Predict Parkinson's from an uploaded image using the selected model."""
    logger.info(f"Received prediction request: file={file.filename}, model_name={model_name}")
    # Validate model name
    if model_name not in METRICS:
        logger.warning(f"Invalid model requested: {model_name}")
        raise HTTPException(status_code=400, detail=f"Invalid model_name. Choose from: {list(METRICS.keys())}")
    pickle_file = METRICS[model_name]["pickle_file"]
    # Validate file type (case-insensitive)
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        logger.warning(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PNG and JPG images are supported.")
    # Save uploaded file securely
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
    temp_path = os.path.join(TEMP_UPLOAD_DIR, temp_filename)
    try:
        logger.info(f"Saving uploaded file to {temp_path}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_size = os.path.getsize(temp_path)
        logger.info(f"Saved file size: {file_size} bytes")
        # Validate image with PIL
        try:
            with Image.open(temp_path) as img:
                img.verify()
            logger.info(f"PIL successfully verified image: {temp_path}")
        except Exception as pil_err:
            logger.error(f"Uploaded file is not a valid image: {pil_err}")
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
        # Run prediction
        logger.info(f"Running prediction using model: {pickle_file}")
        predictor = Predict(model_name=pickle_file.replace("model_", "").replace(".pkl", ""))
        result = predictor.predict_from_image(temp_path)
        prediction = result[0].capitalize()
        logger.info(f"Prediction successful for {file.filename} with model {pickle_file}: {prediction}")
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
    finally:
        # Clean up temp file
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info(f"Cleaned up temp file: {temp_path}")
        except Exception as cleanup_err:
            logger.warning(f"Failed to clean up temp file: {cleanup_err}")
