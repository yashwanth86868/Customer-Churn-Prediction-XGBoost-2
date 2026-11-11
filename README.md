# Customer Churn Prediction Using XGBoost

An end-to-end Machine Learning application that predicts whether a bank customer is likely to leave the bank using an XGBoost classification model.

## Project Overview

Customer churn is an important challenge for banks. This project uses customer information to predict the likelihood of customer attrition and helps identify customers who may need retention strategies.

The trained model is deployed through a FastAPI backend and a Streamlit frontend.

## Live Demo

- **Frontend:** https://customer-churn-prediction-xgboost-2-1.onrender.com
- **Backend API:** https://customer-churn-prediction-xgboost-2.onrender.com
- **API Documentation:** https://customer-churn-prediction-xgboost-2.onrender.com/docs

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- FastAPI
- Streamlit
- Docker
- Pytest
- GitHub Actions
- Render Cloud

## Machine Learning Workflow

1. Load and understand the customer churn dataset.
2. Perform data preprocessing.
3. Encode categorical features using OneHotEncoder.
4. Split data into training and testing sets.
5. Train the XGBoost classification model.
6. Evaluate model performance.
7. Save the trained preprocessing and model pipeline using Pickle.

## Application Architecture

```text
User
  |
  v
Streamlit Frontend
  |
  v
FastAPI REST API
  |
  v
Preprocessing Pipeline
  |
  v
XGBoost Classifier
  |
  v
Churn Prediction
```

## Features

- Customer information input form.
- Real-time prediction through REST API.
- Customer stay or leave prediction.
- Churn probability estimation.
- Cloud-hosted frontend and backend.

## Run Locally

Clone the repository:

```bash
git clone https://github.com/yashwanth86868/Customer-Churn-Prediction-XGBoost-2.git
```

Navigate to the project directory:

```bash
cd Customer-Churn-Prediction-XGBoost-2
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start FastAPI:

```bash
uvicorn main:app --reload
```

Start Streamlit in another terminal:

```bash
streamlit run app.py
```

Open the application at:

`http://localhost:8501`

API documentation:

`http://localhost:8000/docs`

## Automated Testing and CI

This project uses Pytest for automated API testing and GitHub Actions for Continuous Integration.

Tests run automatically when code is pushed to the main branch or a pull request is created.

## Docker

The project includes a Dockerfile and Compose configuration for containerized execution.

## Future Improvements

- Model performance monitoring.
- Input validation and improved error handling.
- Model explainability using SHAP.
- Model versioning and experiment tracking.
- Enhanced deployment automation.

## Author

Yashwanth Balija

GitHub: https://github.com/yashwanth86868
