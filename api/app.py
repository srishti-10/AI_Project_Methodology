"""
FastAPI for Customer Churn Prediction Model.

This module provides a REST API for making churn predictions
using the trained model.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from models.predict import ChurnPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using machine learning",
    version="1.0.0"
)

# Pydantic models for request/response
class CustomerData(BaseModel):
    tenure: float
    numberofdeviceregistered: int
    satisfactionscore: int
    daysincelastorder: int
    cashbackamount: float
    preferedordercat: str
    maritalstatus: str
    complain: int

class PredictionRequest(BaseModel):
    data: List[CustomerData]

class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    churn_risk: List[str]
    model_info: Dict[str, Any]

# Initialize predictor
try:
    predictor = ChurnPredictor()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    predictor = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make churn predictions for multiple customers.
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert to list of dictionaries
        input_data = [customer.dict() for customer in request.data]
        
        # Make predictions
        results = predictor.predict(input_data)
        
        return PredictionResponse(
            predictions=results['predictions'],
            probabilities=results['probabilities'],
            churn_risk=results['churn_risk'],
            model_info=predictor.get_model_performance()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_single")
async def predict_single(customer: CustomerData):
    """
    Make churn prediction for a single customer.
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert to list for prediction
        input_data = [customer.dict()]
        
        # Make prediction
        results = predictor.predict(input_data)
        
        return {
            "prediction": results['predictions'][0],
            "probability": results['probabilities'][0],
            "churn_risk": results['churn_risk'][0],
            "customer_data": customer.dict()
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_info")
async def model_info():
    """Get model information and performance metrics."""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        return predictor.get_model_performance()
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sample_data")
async def get_sample_data():
    """Get sample customer data for testing."""
    sample_data = [
        {
            "tenure": 12.0,
            "numberofdeviceregistered": 3,
            "satisfactionscore": 4,
            "daysincelastorder": 15,
            "cashbackamount": 45.5,
            "preferedordercat": "Electronics",
            "maritalstatus": "Married",
            "complain": 0
        },
        {
            "tenure": 6.0,
            "numberofdeviceregistered": 1,
            "satisfactionscore": 2,
            "daysincelastorder": 90,
            "cashbackamount": 10.0,
            "preferedordercat": "Clothing",
            "maritalstatus": "Single",
            "complain": 1
        }
    ]
    
    return {"sample_data": sample_data}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 