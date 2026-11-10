
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from pathlib import Path




import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

print("Loading model from:", MODEL_PATH)

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Run train_model.py first."
    )

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

print("Model loaded successfully!")

# Create FastAPI application
app = FastAPI(title="Customer Churn Prediction API")

# Load trained model
MODEL_PATH = Path(__file__).resolve().parent / "models" / "churn_model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# Define customer input data
class CustomerData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


# Home endpoint
@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


# Prediction endpoint
@app.post("/predict")
def predict_churn(customer: CustomerData):

    # Convert input data to DataFrame
    input_data = pd.DataFrame([customer.model_dump()])

    # Predict customer churn
    prediction = int(model.predict(input_data)[0])

    # Predict probability
    probability = float(model.predict_proba(input_data)[0][1])

    return {
        "prediction": prediction,
        "churn_probability": round(probability, 4),
        "result": "Customer may leave" if prediction == 1 else "Customer may stay"
    }
