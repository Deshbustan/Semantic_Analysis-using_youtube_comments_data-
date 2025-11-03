# deployment/app.py

import os
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# =====================================================================
# Set the Model URI
# =====================================================================
# This URI points to the model inside the 'mlruns' directory that
# will be copied into our Docker container.
MODEL_URI = "runs:/6ac887c8138146f09dfab753ac55c649/model" # <-- REPLACE THIS!
# =====================================================================

# Load the model at application startup
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize the FastAPI app
app = FastAPI(
    title="YouTube Sentiment API",
    description="An API to predict sentiment of YouTube comments using an MLflow model.",
    version="1.0"
)

# --- Define the structure of the request and response using Pydantic ---

# This defines what a single data point should look like.
# For a sentiment model, it's likely a text comment.
# Update the field name "comment" to match your model's expected feature name.
class Comment(BaseModel):
    comment: str

# This defines the structure of the incoming request list
class PredictionRequest(BaseModel):
    data: List[Comment]

# --- Create the prediction endpoint ---

@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Receives a list of comments and returns sentiment predictions.
    - **request**: A JSON object containing a 'data' key with a list of comment objects.
    - **returns**: A JSON object with the model's predictions.
    """
    try:
        # Convert Pydantic models to a list of dictionaries
        input_data = [item.dict() for item in request.data]
        
        # MLflow's pyfunc predict() expects a pandas DataFrame
        data_df = pd.DataFrame(input_data)

        # Get predictions
        predictions = model.predict(data_df)

        # Return predictions. FastAPI automatically converts this to JSON.
        return {"predictions": predictions.tolist()}

    except Exception as e:
        # If something goes wrong, return an HTTP exception
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API. Go to /docs for the interactive API documentation."}