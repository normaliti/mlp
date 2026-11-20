import json
from pathlib import Path

def save_history(path, model_name, model, config, history):
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    serial_history = {}
    for metric_name, metric_val in history.items():
        serial_history[metric_name] = [float(value) for value in metric_val]

    history_data = {
        "name": model_name,
        "model": {
            "layer_sizes": [int(size) for size in model.layer_sizes],
            "activation": model.activation
        },
        "training": {
            "epochs_requested": config.epochs,
            "epochs_completed": len(serial_history["train_loss"]),
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "early_stopping": config.early_stopping
        },
        "history": serial_history
    }
    with history_path.open("w", encoding="utf-8") as history_file:
        json.dump(history_data, history_file, indent=4)
    print(f"history saved to: {history_path}")

def load_history(path):
    history_path = Path(path)
    if not history_path.is_file():
        raise FileNotFoundError(f"History file not found: {history_path}")
    try:
        with history_path.open("r", encoding="utf-8") as history_file:
            return json.load(history_file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid history JSON: {history_path}") from error
