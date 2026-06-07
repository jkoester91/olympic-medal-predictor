import mlflow
from mlflow.tracking import MlflowClient

def get_best_run_id(experiment_name="Olympic_Medal_Prediction", metric="metrics.f1"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"{metric} DESC"]
    )
    return runs[0].info.run_id

if __name__ == "__main__":
    best_id = get_best_run_id()
    print(f"The best performing run is: {best_id}")