import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import yaml
import pathlib
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import preprocess_pipeline

def get_best_model_uri():
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Olympic_Medal_Prediction")
    
    # Guard against experiment not existing
    if experiment is None:
        raise ValueError("Experiment 'Olympic_Medal_Prediction' not found. Have you run train.py?")
        
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], 
        order_by=["metrics.roc_auc DESC"], 
        max_results=1
    )
    if not runs: 
        raise FileNotFoundError("No runs found in experiment.")
    return f"runs:/{runs[0].info.run_id}/model"

def run_evaluation():
    # 1. Configure the environment globally FIRST
    tracking_uri = pathlib.Path(os.path.abspath("mlruns")).as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    
    with open("configs/config.yaml", "r") as f: 
        config = yaml.safe_load(f)
    
    # 2. Preprocess and copy to ensure clean memory
    _, X_test_raw, _, y_test = preprocess_pipeline(
        athlete_path=config["data"]["raw_athlete_path"],
        noc_path=config["data"]["raw_noc_path"],
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )
    # Ensure memory safety with .copy()
    X_test = X_test_raw[['Age', 'Height', 'Weight']].copy()

    # 3. Load & Predict
    model_uri = get_best_model_uri()
    model = mlflow.sklearn.load_model(model_uri)
    probs = model.predict_proba(X_test)[:, 1]
    
    # Apply custom threshold
    threshold = 0.55
    y_pred = (probs >= threshold).astype(int)
    
    # 4. Report
    print("\n" + "="*50)
    print("FINAL MODEL PERFORMANCE REPORT")
    print("="*50)
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred, target_names=["No Medal", "Medal Won"]))

if __name__ == "__main__":
    run_evaluation()