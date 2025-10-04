import os
import json

def save_model_metrics(display_name, metrics, model_path, metrics_file=None):
    """
    Save or update metrics for a model in a consolidated metrics.json file.
    Args:
        display_name (str): The display name for the model (used as the key in metrics.json).
        metrics (dict): Dictionary of metrics to save.
        model_path (str): Path to the model pickle file (or just the file name).
        metrics_file (str, optional): Path to the consolidated metrics file. Defaults to 'models/metrics.json'.
    """
    if metrics_file is None:
        metrics_file = os.path.join("models", "metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}
    metrics_to_save = dict(metrics)
    metrics_to_save["pickle_file"] = os.path.basename(model_path)
    all_metrics[display_name] = metrics_to_save
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[INFO] Metrics saved/updated in {metrics_file}")
