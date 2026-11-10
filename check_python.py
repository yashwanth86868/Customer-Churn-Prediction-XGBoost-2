import os

file_path = "models/churn_model.pkl"

if os.path.exists(file_path):
    print("File exists!")
    print("File size:", os.path.getsize(file_path), "bytes")
else:
    print("Model file not found!")