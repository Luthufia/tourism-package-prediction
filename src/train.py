# ============================================================
# Tourism Package Prediction - Model Training Pipeline
# ============================================================

import os
import pandas as pd

# Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Modeling
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Serialization
import joblib

# Hugging Face
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# MLflow
import mlflow

# ============================================================
# Hugging Face Authentication
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

# ============================================================
# MLflow Setup
# ============================================================

mlflow.set_experiment("MLOps_Tourism")

# ============================================================
# Load Data from Hugging Face Dataset Hub
# ============================================================

X_train = pd.read_csv(
    "https://huggingface.co/datasets/Luthufia/tourism-package-prediction/resolve/main/X_train.csv"
)

X_test = pd.read_csv(
    "https://huggingface.co/datasets/Luthufia/tourism-package-prediction/resolve/main/X_test.csv"
)

y_train = pd.read_csv(
    "https://huggingface.co/datasets/Luthufia/tourism-package-prediction/resolve/main/y_train.csv"
).squeeze()

y_test = pd.read_csv(
    "https://huggingface.co/datasets/Luthufia/tourism-package-prediction/resolve/main/y_test.csv"
).squeeze()

print("Datasets loaded successfully.")

# ============================================================
# Feature Lists
# ============================================================

numeric_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]

# ============================================================
# Handle Class Imbalance
# ============================================================

class_weight = (
    y_train.value_counts()[0]
    /
    y_train.value_counts()[1]
)

print(f"Scale Positive Weight: {class_weight:.2f}")

# ============================================================
# Preprocessing Pipeline
# ============================================================

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# ============================================================
# XGBoost Model
# ============================================================

xgb_model = xgb.XGBClassifier(
    random_state=42,
    scale_pos_weight=class_weight,
    eval_metric="logloss",
)

# ============================================================
# Hyperparameter Grid
# ============================================================

xgb_param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5, 7, 9],
    "xgbclassifier__learning_rate": [0.01, 0.05],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.8, 1.0],
}

# ============================================================
# Model Pipeline
# ============================================================

xgb_pipeline = make_pipeline(
    preprocessor,
    xgb_model,
)

# ============================================================
# Training + MLflow Tracking
# ============================================================

with mlflow.start_run(run_name="XGBoost"):

    grid_search = GridSearchCV(
        xgb_pipeline,
        xgb_param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    results = grid_search.cv_results_

    for i in range(len(results["params"])):

        with mlflow.start_run(
            nested=True,
            run_name=f"XGB_Trial_{i+1}",
        ):

            mlflow.log_params(
                results["params"][i]
            )

            mlflow.log_metric(
                "mean_test_score",
                results["mean_test_score"][i],
            )

            mlflow.log_metric(
                "std_test_score",
                results["std_test_score"][i],
            )

    mlflow.log_params(
        grid_search.best_params_
    )

    best_model = grid_search.best_estimator_

    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    train_report = classification_report(
        y_train,
        y_pred_train,
        output_dict=True,
    )

    test_report = classification_report(
        y_test,
        y_pred_test,
        output_dict=True,
    )

    mlflow.log_metrics(
        {
            "train_accuracy": train_report["accuracy"],
            "train_precision": train_report["1"]["precision"],
            "train_recall": train_report["1"]["recall"],
            "train_f1_score": train_report["1"]["f1-score"],
            "test_accuracy": test_report["accuracy"],
            "test_precision": test_report["1"]["precision"],
            "test_recall": test_report["1"]["recall"],
            "test_f1_score": test_report["1"]["f1-score"],
        }
    )

    model_path = "tourism_xgboost_model.joblib"

    joblib.dump(
        best_model,
        model_path,
    )

    mlflow.log_artifact(
        model_path,
        artifact_path="model",
    )

    print(f"Model saved: {model_path}")

    repo_id = "Luthufia/tourism-xgboost-model"

    try:
        api.repo_info(
            repo_id=repo_id,
            repo_type="model",
        )
        print("Model repository exists.")

    except RepositoryNotFoundError:

        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=False,
            token=HF_TOKEN,
        )

        print("Model repository created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type="model",
    )

    print("Model uploaded to Hugging Face.")

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nTest Accuracy:")
    print(round(test_report["accuracy"], 4))

    print("\nTest F1 Score:")
    print(round(test_report["1"]["f1-score"], 4))

print("Training pipeline completed successfully.")
