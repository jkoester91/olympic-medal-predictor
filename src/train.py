import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 0. Global Configuration
mlflow.set_tracking_uri(f"file:{os.path.abspath('mlruns')}")
mlflow.set_experiment("Olympic_Medal_Prediction")

# 1. Load and Prep Data
df = pd.read_csv("data/athlete_events.csv")
df['medal_won'] = df['Medal'].notna().astype(int)
features = ['Age', 'Height', 'Weight']
df_clean = df.dropna(subset=features).copy()
X = df_clean[features]
y = df_clean['medal_won']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Splt the remaining 90% into Train (80%) and Val (20 OF THE 90%)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# 2. Model Training Loop
configs = {
    "baseline_rf": {"type": "rf", "max_depth": 10, "n_estimators": 50},
    "deep_rf": {"type": "rf", "max_depth": 20, "n_estimators": 100},
    "wide_rf": {"type": "rf", "max_depth": 12, "n_estimators": 200},
    "gradient_boost": {"type": "gb", "max_depth": 3, "n_estimators": 50, "learning_rate": 0.1},
    "fast_gb": {"type": "gb", "max_depth": 5, "n_estimators": 100, "learning_rate": 0.2}
}

for run_name, params in configs.items():
    with mlflow.start_run(run_name=run_name):
        # Filter out the routing key
        model_params = {k: v for k, v in params.items() if k != "type"}
        mlflow.log_params(model_params)
        
        if params["type"] == "rf":
            # Unpack model_params using **
            model = RandomForestClassifier(**model_params, class_weight="balanced", random_state=42)
            model.fit(X_train, y_train)
        else:
            # Unpack model_params using **
            model = GradientBoostingClassifier(**model_params, random_state=42)
            weights = compute_sample_weight(class_weight="balanced", y=y_train)
            model.fit(X_train, y_train, sample_weight=weights)

        # Evaluate and log metrics
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_val, preds),
            "precision": precision_score(y_val, preds, zero_division=0),
            "recall": recall_score(y_val, preds, zero_division=0),
            "f1": f1_score(y_val, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_val, probs)
        }
        
        # Log signature for contract-based evaluation
        signature = infer_signature(X_train, preds)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)
        print(f"✅ Logged {run_name} metrics and artifacts.")