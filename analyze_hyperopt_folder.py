# Check each tuned result from keras-tuner-logs
import os
import json

def get_best_trial(folder, metric = "val_loss", metric_mode = "min"):
    # Get all the log directories
    log_dirs = os.listdir(folder)

    model_metrics = []
    trial_jsons = []
    # Check the accuracy and loss for each trial.json file (one in each log directory)
    for log_dir in log_dirs:
        if not os.path.isdir(f'{folder}/{log_dir}'):
            continue
        with open(f'{folder}/{log_dir}/trial.json') as f:
            trial_json = json.load(f)
        trial_jsons.append(trial_json)
        model_metric = trial_json['metrics']["metrics"][metric]["observations"][0]["value"][0]
        model_metrics.append(model_metric)
    
    # Find the best trial
    if metric_mode == "min":
        best_metric = min(model_metrics)
    elif metric_mode == "max":
        best_metric = max(model_metrics)
    else:
        raise ValueError(f"metric_mode must be either 'min' or 'max', but was {metric_mode}")
    best_metric_idx = model_metrics.index(best_metric)
    best_trial = trial_jsons[best_metric_idx]
    return best_trial, best_metric
