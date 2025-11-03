import mlflow
from pathlib import Path

mlruns_folder = Path("C:\Coding\AI_ML\YouTube Semantic Analysis\mlflow\mlflow_data").resolve()
tracking_uri = f"sqlite:///{mlruns_folder.joinpath('mlflow2.db')}"
mlflow.set_tracking_uri(tracking_uri)

print(f"Successfully connected to MLflow tracking URI: {mlflow.get_tracking_uri()}")

experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"Experiment: name={exp.name}, ID={exp.experiment_id}")

mlflow.set_tracking_uri("http://127.0.0.1:5000")

print(f"Connected to MLflow at: {mlflow.get_tracking_uri()}")

try:
    # Search all runs in the default experiment '0'
    runs_df = mlflow.search_runs(experiment_ids=['0'])

    # Display the runs so you can see them
    print("\n--- Found the following runs ---")
    # You might want to view other columns like 'metrics.f1_score', etc.
    print(runs_df[['run_id', 'metrics.accuracy', 'status']])

    # --- CHOOSE YOUR MODEL ---
    # Find the run with the highest accuracy
    if 'metrics.accuracy' in runs_df.columns and not runs_df['metrics.accuracy'].isnull().all():
        best_run = runs_df.loc[runs_df['metrics.accuracy'].idxmax()]
        chosen_run_id = best_run['run_id']
        print(f"\nAutomatically selected best run with ID: {chosen_run_id}")
    else:
        # If you can't find by accuracy, manually pick one from the table above
        chosen_run_id = input("\nPlease enter the run_id you want to package: ")

    # Construct the model URI. 'model' is the default name MLflow uses when saving models.
    model_uri = f"runs:/{chosen_run_id}/model"

    print("\n" + "="*50)
    print(f"Your Model URI is: {model_uri}")
    print("="*50)
    print("Copy this URI for the next step.")

except Exception as e:
    print(f"\nAn error occurred: {e}")
