import os
import pandas as pd

from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# --------------------------------------------------
# Hugging Face Authentication
# --------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

# --------------------------------------------------
# Load Dataset from Hugging Face
# --------------------------------------------------

DATASET_PATH = "hf://datasets/Luthufia/tourism-package-prediction/tourism.csv"

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")

# --------------------------------------------------
# Data Cleaning
# --------------------------------------------------

df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True)

df["Gender"] = df["Gender"].replace(
    {"Fe Male": "Female"}
)

df["MaritalStatus"] = df["MaritalStatus"].replace(
    {"Unmarried": "Single"}
)

print("Data cleaning completed.")

# --------------------------------------------------
# Feature and Target Separation
# --------------------------------------------------

X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"]

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Train-test split completed.")

# --------------------------------------------------
# Save Files Locally
# --------------------------------------------------

os.makedirs("data", exist_ok=True)

X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)

y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("Files saved locally.")

# --------------------------------------------------
# Upload Files to Hugging Face
# --------------------------------------------------

files_to_upload = [
    "data/X_train.csv",
    "data/X_test.csv",
    "data/y_train.csv",
    "data/y_test.csv"
]

for file_path in files_to_upload:

    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id="Luthufia/tourism-package-prediction",
        repo_type="dataset"
    )

    print(f"Uploaded: {os.path.basename(file_path)}")

print("Data preparation completed successfully.")
