import os
import shutil
import json
import logging
import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.models import ModelsResponse, PredictionResponse
from src.api.endpoints import router as api_router

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("parkinsons_api")

# Environment-based config
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost,http://127.0.0.1").split(",")
TEMP_UPLOAD_DIR = os.getenv("TEMP_UPLOAD_DIR", "temp_uploads")

# Path to models and metrics
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models'))
METRICS_PATH = os.path.join(MODELS_DIR, 'metrics.json')

# FastAPI app instance
app = FastAPI(title="Parkinson's Detection API", description="API for Parkinson's Disease Detection", version="1.0.0")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes from endpoints.py
app.include_router(api_router)
