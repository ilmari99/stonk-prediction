# Check each tuned result from keras-tuner-logs
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

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
        print(f"Trial {trial_json}")
        try:
            model_metric = trial_json['metrics']["metrics"][metric]["observations"][0]["value"][0]
        except KeyError:
            print(f"Could not find metric {metric} in {trial_json}")
            continue
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

def hyperopt_results_to_dataframe(folder):
    """ Read the results from a hyperopt run and convert to a dataframe """
    df = None
    log_dirs = os.listdir(folder)
    hyperparams = []
    # Check the accuracy and loss for each trial.json file (one in each log directory)
    for log_dir in log_dirs:
        if not os.path.isdir(f'{folder}/{log_dir}'):
            continue
        with open(f'{folder}/{log_dir}/trial.json') as f:
            trial_json = json.load(f)
        if trial_json["status"] != "COMPLETED":
            continue
        #if not hyperparams:
        hyperparams = list(trial_json["hyperparameters"]["values"].keys())
                
        trial_id = trial_json["trial_id"]
        # Add the hp values to the dataframe
        values = list(trial_json["hyperparameters"]["values"].values())
        val_loss = trial_json['metrics']["metrics"]["val_loss"]["observations"][0]["value"][0]
        val_accuracy = trial_json['metrics']["metrics"]["val_multi_accuracy"]["observations"][0]["value"][0]
        profit = trial_json['metrics']["metrics"]["profit"]["observations"][0]["value"][0]
        # Add the values to the dataframe
        if df is None:
            df = pd.DataFrame(columns = ["trial_id"] + hyperparams + ["val_loss", "val_multi_accuracy", "profit"])
        df = pd.concat([df, pd.DataFrame([[trial_id] + values + [val_loss, val_accuracy,profit]], columns = ["trial_id"] + hyperparams + ["val_loss", "val_multi_accuracy", "profit"])])
    return df
        
        
        
        
if __name__ == "__main__":
    folder = "/home/ilmari/python/stonk-prediction/keras-tuner-dir-lstm2/direction_prediction_48_val_loss"
    best_trial, best_metric = get_best_trial(folder)
    print(f"Best trial: {best_trial}")
    print(f"Best metric: {best_metric}")
    df = hyperopt_results_to_dataframe(folder)
    print(df.columns)
    print(df)
    # Plot the distribution of the validation loss
    plt.hist(df["val_loss"], bins = 20, label="val_loss")
    
    # Plot the distribution of the validation accuracy
    plt.hist(df["val_multi_accuracy"], bins = 20, label="val_accuracy")
    plt.title("Distribution of validation loss and accuracy in different training runs")
    plt.legend()
    
    # Plot the correlation between columns and the validation loss
    fig, axes = plt.subplots(2,2)
    ax_row_index = 0
    ax_col_index = 0
    for col in df.columns:
        if col in ["trial_id", "val_loss", "val_multi_accuracy", "tuner/bracket", "tuner/round","tuner/epochs", "tuner/trial_id"]:
            continue
        ax = axes[ax_row_index, ax_col_index]
        print(f"Plotting {col}")
        ax.scatter(df[col], df["val_loss"])
        # Show the confidence interval
        #ax.axvline(best_trial["hyperparameters"]["values"][col], color = "red", label = "Best value")
        ax.set_title(f"{col} vs. val_loss")
        ax.set_xlabel(col)
        ax.set_ylabel("val_loss")
        ax.grid()
        ax_col_index += 1
        if ax_col_index >= 2:
            ax_col_index = 0
            ax_row_index += 1
        if ax_row_index >= 2:
            break
    fig.suptitle("Correlation between hyperparameters and validation loss")
    plt.tight_layout()

    # Find the row with the lowest val_loss
    best_row = df.iloc[df["val_loss"].idxmin()]
    print(f"Best row: {best_row}")

    plt.show()
