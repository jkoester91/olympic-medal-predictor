import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ==========================================
# 1. BULLETPROOF DATA LOADING & SPLITTING
# ==========================================
print("Loading and preparing data from athlete_events.csv...")

# Load the raw dataset
df = pd.read_csv("data/athlete_events.csv")

# Create a binary target: 1 if they won ANY medal, 0 if NaN (no medal)
df['medal_won'] = df['Medal'].notna().astype(int)

# Select numeric features for our model
features = ['Age', 'Height', 'Weight']

# Drop rows with missing values in these specific columns so the models don't crash
df_clean = df.dropna(subset=features).copy()

X = df_clean[features]
y = df_clean['medal_won']

# Split the data into Training (80%) and Validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data ready! Training on {len(X_train)} athletes, Validating on {len(X_val)} athletes.")
# ==========================================


# ==========================================
# 2. MODEL CONFIGURATIONS
# ==========================================
configs = {
    "baseline_rf": {"type": "rf", "max_depth": 10, "n_estimators": 50},
    "deep_rf": {"type": "rf", "max_depth": 20, "n_estimators": 100},
    "wide_rf": {"type": "rf", "max_depth": 12, "n_estimators": 200},
    "gradient_boost": {"type": "gb", "max_depth": 3, "n_estimators": 50, "learning_rate": 0.1},
    "fast_gb": {"type": "gb", "max_depth": 5, "n_estimators": 100, "learning_rate": 0.2}
}

mlflow.set_experiment("Olympic_Medal_Prediction")

for run_name, params in configs.items():
    print(f"Starting training run: {run_name}...")
    
    with mlflow.start_run(run_name=run_name):
        
        # Initialize the models with Class Balancing
        if params["type"] == "rf":
            model = RandomForestClassifier(
                max_depth=params["max_depth"],
                n_estimators=params["n_estimators"],
                class_weight="balanced",  # <-- Penalizes missing medal winners
                random_state=42
            )
            mlflow.log_params({
                "max_depth": params["max_depth"],
                "n_estimators": params["n_estimators"],
                "class_weight": "balanced"
            })
            model.fit(X_train, y_train)
            
        elif params["type"] == "gb":
            model = GradientBoostingClassifier(
                max_depth=params["max_depth"],
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                random_state=42
            )
            mlflow.log_params({
                "max_depth": params["max_depth"],
                "n_estimators": params["n_estimators"],
                "learning_rate": params["learning_rate"]
            })
            # Compute sample weights to handle imbalance in Gradient Boosting
            sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
            model.fit(X_train, y_train, sample_weight=sample_weights)

        # Evaluate the models
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else preds
        
        metrics = {
            "accuracy": accuracy_score(y_val, preds),
            "precision": precision_score(y_val, preds, zero_division=0),
            "recall": recall_score(y_val, preds, zero_division=0),
            "f1": f1_score(y_val, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_val, probs)
        }
        
        # Log metrics & model artifact to MLflow
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")
        
        print(f"    ✅ Logged metrics & artifacts for {run_name}.")