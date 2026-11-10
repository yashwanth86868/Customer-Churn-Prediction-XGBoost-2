
# Customer Churn Prediction using XGBoost

# 1. Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

from xgboost import XGBClassifier


# 2. Load dataset
dataset = pd.read_csv(
    r"C:\Users\ybali\Downloads\Churn_Modelling.csv"
)

# 3. Independent and dependent variables
X = dataset.iloc[:, 3:-1]
y = dataset.iloc[:, -1]

# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=0,
    stratify=y
)

# 5. Encode categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"), [1, 2])
    ],
    remainder="passthrough"
)

# 6. Create XGBoost classifier
classifier = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    eval_metric="logloss",
    random_state=0
)

# 7. Create machine learning pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", classifier)
])

# 8. Train the model
model.fit(X_train, y_train)





import pickle
from pathlib import Path

# Find the folder containing train_model.py
BASE_DIR = Path(__file__).resolve().parent

# Create models folder if it doesn't exist
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Define model file path
MODEL_PATH = MODEL_DIR / "churn_model.pkl"

# Save trained model
with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
print("Saved at:", MODEL_PATH)
print("File size:", MODEL_PATH.stat().st_size, "bytes")

print("Model training completed!")

# 9. Predict test data
y_pred = model.predict(X_test)

# 10. Evaluate model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 11. Training and testing accuracy
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print("\nTraining Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# 12. Cross-validation
accuracies = cross_val_score(
    estimator=model,
    X=X_train,
    y=y_train,
    cv=5,
    scoring="accuracy"
)

print("\nCross Validation Results:")
print("Mean Accuracy: {:.2f}%".format(accuracies.mean() * 100))
print("Standard Deviation: {:.2f}%".format(accuracies.std() * 100))



import pickle

# Save the trained model
with open("churn_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")



import pickle

with open("churn_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

print("Model loaded successfully!")


predictions = loaded_model.predict(X_test)

print(predictions[:10])
