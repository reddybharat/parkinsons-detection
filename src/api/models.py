from pydantic import BaseModel
from typing import Dict, Optional

class ModelMetrics(BaseModel):
    """Metrics for a trained model."""
    accuracy: float
    precision: float
    recall: float
    f1_score: Optional[float] = None
    name: Optional[str] = None

class ModelsResponse(BaseModel):
    """Response model for listing available models and their metrics."""
    models: Dict[str, ModelMetrics]

class PredictionResponse(BaseModel):
    """Response model for a prediction result."""
    prediction: str
