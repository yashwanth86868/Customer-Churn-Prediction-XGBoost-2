
import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Application title
st.title("Customer Churn Prediction System")
st.write("Predict whether a bank customer is likely to leave.")

st.divider()

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/predict"

# Customer input form
st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input(
        "Credit Score", min_value=300, max_value=900, value=650
    )

    geography = st.selectbox(
        "Geography", ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender", ["Male", "Female"]
    )

    age = st.number_input(
        "Age", min_value=18, max_value=100, value=35
    )

    tenure = st.number_input(
        "Tenure", min_value=0, max_value=10, value=5
    )

with col2:
    balance = st.number_input(
        "Balance", min_value=0.0, value=50000.0
    )

    products = st.number_input(
        "Number of Products", min_value=1, max_value=4, value=2
    )

    credit_card = st.selectbox(
        "Has Credit Card", [1, 0]
    )

    active_member = st.selectbox(
        "Is Active Member", [1, 0]
    )

    salary = st.number_input(
        "Estimated Salary", min_value=0.0, value=60000.0
    )

st.divider()

# Prediction button
if st.button("Predict Customer Churn", type="primary"):

    # Prepare customer information
    customer_data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": credit_card,
        "IsActiveMember": active_member,
        "EstimatedSalary": salary
    }

    try:
        # Send request to FastAPI
        response = requests.post(
            API_URL,
            json=customer_data,
            timeout=30
        )

        # Check API response
        response.raise_for_status()

        # Get prediction result
        result = response.json()

        prediction = result["prediction"]
        probability = result["churn_probability"]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("Customer may leave the bank!")
        else:
            st.success("Customer is likely to stay!")

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. "
            "Please start the backend server."
        )

    except requests.exceptions.RequestException as error:
        st.error(f"API request failed: {error}")

    except (KeyError, ValueError) as error:
        st.error(f"Invalid API response: {error}")
