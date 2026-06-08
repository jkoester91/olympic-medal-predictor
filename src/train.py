import os
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from pathlib import Path

# 0. Global Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
MLRUNS_PATH = BASE_DIR / "mlruns"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri(MLRUNS_PATH.as_uri())
mlflow.set_experiment("Olympic_Medal_Prediction")

# 1. Load and Prep Data
df = pd.read_csv("data/athlete_events.csv")
df['medal_won'] = df['Medal'].notna().astype(int)
# IMPORTANT: Include 'Sport' in your features now!
features = ['Age', 'Height', 'Weight', 'Sport']
df_clean = df.dropna(subset=features).copy()
X = df_clean[features]
y = df_clean['medal_won']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# 2. Define Preprocessing (Consistent for all models)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['Age', 'Height', 'Weight']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Sport'])
    ]
)

# 3. Model Training Loop
configs = {
    "baseline_rf": {"type": "rf", "max_depth": 10, "n_estimators": 50},
    "deep_rf": {"type": "rf", "max_depth": 20, "n_estimators": 100},
    "wide_rf": {"type": "rf", "max_depth": 12, "n_estimators": 200},
    "gradient_boost": {"type": "gb", "max_depth": 3, "n_estimators": 50, "learning_rate": 0.1},
    "fast_gb": {"type": "gb", "max_depth": 5, "n_estimators": 100, "learning_rate": 0.2}
}

for run_name, params in configs.items():
    with mlflow.start_run(run_name=run_name):
        model_params = {k: v for k, v in params.items() if k != "type"}
        mlflow.log_params(model_params)
        
        # Define the base classifier
        if params["type"] == "rf":
            clf = RandomForestClassifier(**model_params, class_weight="balanced", random_state=42)
        else:
            clf = GradientBoostingClassifier(**model_params, random_state=42)
            
        # Wrap in Pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        # Train
        model.fit(X_train, y_train)

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
        
        signature = infer_signature(X_train, preds)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)
        print(f"✅ Logged {run_name} metrics and artifacts.")