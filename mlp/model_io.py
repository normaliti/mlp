from pathlib import Path
import numpy as np
from mlp.model import MLP

CLASS_NAMES = np.array(["B", "M"])

def save_model(path, model, mean, std):
    model_path = Path(path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_data = {"layer_sizes": np.array(model.layer_sizes, dtype=int),
                  "activation": np.array(model.activation),
                  "mean": mean, "std": std, "class_names": CLASS_NAMES}
    for wlayer_idx in range(len(model.weights)):
        model_data[f"weight_{wlayer_idx}"] = model.weights[wlayer_idx]
        model_data[f"biases_{wlayer_idx}"] = model.biases[wlayer_idx]
    np.savez_compressed(model_path, **model_data)
    print(f"model saved to: {model_path}")

def load_model(path):
    model_path = Path(path)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with np.load(model_path, allow_pickle=False) as model_data:
        layer_sizes = model_data["layer_sizes"].astype(int).tolist()
        activation = str(model_data["activation"].item())
        mean = model_data["mean"].copy()
        std = model_data["std"].copy()
        class_names = model_data["class_names"].astype(str).tolist()
        model = MLP(layer_sizes=layer_sizes,
                    activation=activation)
        for wlayer_idx in range(len(model.weights)):
            model.weights[wlayer_idx] = model_data[f"weight_{wlayer_idx}"].copy()
            model.biases[wlayer_idx] = model_data[f"biases_{wlayer_idx}"].copy()
    return model, mean, std, class_names
