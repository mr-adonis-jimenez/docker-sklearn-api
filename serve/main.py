"""FastAPI application for scikit-learn model serving."""
import os
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="scikit-learn ML API",
    description="Production-ready ML inference API built with FastAPI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Model configuration
MODEL_PATH = Path("models/model.joblib")
MODEL_VERSION = "1.0.0"
model = None


class PredictionInput(BaseModel):
    """Input schema for predictions."""
    features: List[float] = Field(..., description="Input features for prediction")

    class Config:
        json_schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for predictions."""
    prediction: int = Field(..., description="Predicted class")
    probability: float = Field(..., description="Prediction probability")
    model_version: str = Field(..., description="Model version used")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool


class ModelInfo(BaseModel):
    """Model information response."""
    version: str
    model_type: str
    features_count: int


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.get("/", tags=["Root"])
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "message": "scikit-learn ML API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
    )


@app.get("/model-info", response_model=ModelInfo, tags=["Model"])
async def model_info() -> ModelInfo:
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        version=MODEL_VERSION,
        model_type=type(model).__name__,
        features_count=model.n_features_in_ if hasattr(model, 'n_features_in_') else 0,
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(input_data: PredictionInput) -> PredictionOutput:
    """Make prediction endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to numpy array
        features = np.array([input_data.features])
        
        # Make prediction
        prediction = int(model.predict(features)[0])
        
        # Get prediction probability
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            probability = float(probabilities[prediction])
        else:
            probability = 1.0
        
        return PredictionOutput(
            prediction=prediction,
            probability=probability,
            model_version=MODEL_VERSION,
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction error: {str(e)}"
        )
