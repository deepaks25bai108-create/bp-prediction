"""
train.py
--------
Blood Pressure Prediction ML Pipeline.
Trains a model on clinical and lifestyle risk factors to predict:
  1. Systolic Blood Pressure (SBP)
  2. Diastolic Blood Pressure (DBP)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data.generate_data import generate_blood_pressure_dataset

def load_or_create_data():
    data_path = os.path.join(os.path.dirname(__file__), "data", "bp_data.csv")
    if os.path.exists(data_path):
        print(f"Loading existing dataset from {data_path}...")
        df = pd.read_csv(data_path)
    else:
        print("Dataset not found. Generating fresh clinical dataset...")
        df = generate_blood_pressure_dataset()
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Dataset generated and saved ({len(df)} records).")
    return df

def build_pipeline():
    numeric_features = [
        "Age",
        "BMI",
        "Heart_Rate",
        "Daily_Steps",
        "Sleep_Hours",
        "Stress_Level"
    ]
    categorical_features = [
        "Gender",
        "Smoking_Status",
        "Alcohol_Intake",
        "Salt_Intake"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )
    return pipeline, numeric_features, categorical_features

def evaluate_predictions(y_true, y_pred, split_name="Test"):
    metrics = {}
    targets = ["Systolic_BP", "Diastolic_BP"]
    for i, target in enumerate(targets):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        metrics[target] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"[{split_name}] {target:13s} -> MAE: {mae:.2f} mmHg | RMSE: {rmse:.2f} mmHg | R²: {r2:.3f}")
    return metrics

def train_and_save():
    df = load_or_create_data()

    feature_cols = [
        "Age", "Gender", "BMI", "Heart_Rate", "Daily_Steps",
        "Sleep_Hours", "Stress_Level", "Smoking_Status",
        "Alcohol_Intake", "Salt_Intake"
    ]
    target_cols = ["Systolic_BP", "Diastolic_BP"]

    X = df[feature_cols]
    y = df[target_cols].values

    # Strict featurization ordering: split into train & test BEFORE any preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining set size: {len(X_train)} | Test set size: {len(X_test)}")

    pipeline, num_feats, cat_feats = build_pipeline()

    print("\nTraining Random Forest Multi-Output Model...")
    pipeline.fit(X_train, y_train)

    print("\n--- Model Evaluation ---")
    y_train_pred = pipeline.predict(X_train)
    evaluate_predictions(y_train, y_train_pred, split_name="Train")
    
    y_test_pred = pipeline.predict(X_test)
    test_metrics = evaluate_predictions(y_test, y_test_pred, split_name="Test")

    # Save model and feature metadata
    model_output_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    artifact = {
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "numeric_features": num_feats,
        "categorical_features": cat_feats,
        "target_cols": target_cols,
        "test_metrics": test_metrics
    }
    joblib.dump(artifact, model_output_path)
    print(f"\nModel pipeline successfully saved to: {model_output_path}")

if __name__ == "__main__":
    train_and_save()
