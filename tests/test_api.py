import sys
from pathlib import Path

# Add project root directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from main import app

# Create test client
client = TestClient(app)


# Test 1: Home endpoint
def test_home():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "Customer Churn Prediction API is running"
    }


# Test 2: Prediction endpoint
def test_predict():

    customer_data = {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Male",
        "Age": 35,
        "Tenure": 5,
        "Balance": 50000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 60000
    }

    response = client.post("/predict", json=customer_data)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]
    assert 0 <= result["churn_probability"] <= 1
    assert "result" in result