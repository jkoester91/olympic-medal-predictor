import os
import yaml
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import preprocess_pipeline

def get_best_run_model_uri(mlruns_dir="mlruns"):
    """
    Uses the native MLflow Client to query local tracking logs and 
    programmatically return the URI of the best training run sorted by ROC-AUC.
    """
    mlflow.set_tracking_uri(f"file:{os.path.abspath(mlruns_dir)}")
    client = MlflowClient()
    
    experiments = client.search_experiments()
    if not experiments:
        raise FileNotFoundError(f"No MLflow experiments found in '{mlruns_dir}'.")
        
    # Programmatically filter and sort by performance metrics
    all_runs = client.search_runs(
        experiment_ids=[exp.experiment_id for exp in experiments],
        order_by=["metrics.roc_auc DESC"],
        max_results=1
    )
    
    if not all_runs:
        raise FileNotFoundError("No training runs found inside the experiment folders.")
        
    best_run_id = all_runs[0].info.run_id
    return f"runs:/{best_run_id}/model"

def run_standalone_evaluation():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Re-generating test split from preprocessing pipeline...")
    _, X_test, _, y_test = preprocess_pipeline(
        athlete_path=config["data"]["raw_athlete_path"],
        noc_path=config["data"]["raw_noc_path"],
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    try:
        model_uri = get_best_run_model_uri()
        print(f"Loading best tracked model artifact via native URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print(f"\nError fetching model: {e}")
        return

    print("Executing final model evaluations on test partition...")
    y_pred = model.predict(X_test)
    
    print("\n" + "="*50)
    print("             FINAL MODEL PERFORMANCE REPORT             ")
    print("="*50)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Medal", "Medal Won"], zero_division=0))
    print("="*50)

if __name__ == "__main__":
    run_standalone_evaluation()