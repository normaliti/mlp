import argparse
from mlp.data import load_dataset
from mlp.model_io import load_model
from mlp.preprocessing import (
    dataframe_to_arrays,
    one_hot_encoding,
    standard_scaler,
)
from mlp.train_utils import (
    accuracy,
    binary_cross_entropy,
    confusion_matrix,
    print_conf_matrix,
    print_classification_metrics
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a trained MLP")
    parser.add_argument("model_path")
    parser.add_argument("dataset_path")

    return parser.parse_args()

def main():
    args = parse_arguments()

    model, mean, std, _ = load_model(args.model_path)

    dataframe = load_dataset(args.dataset_path)

    features, labels = dataframe_to_arrays(dataframe)

    expected_features = model.layer_sizes[0]

    if features.shape[1] != expected_features:
        raise ValueError(
            "Invalid number of features: "
            f"expected {expected_features}, "
            f"got {features.shape[1]}"
        )

    if (len(mean) != expected_features or len(std) != expected_features):
        raise ValueError(
            "Model preprocessing parameters "
            "have invalid shapes"
        )

    features = standard_scaler(features, mean, std)
    targets = one_hot_encoding(labels)
    probabilities = model.forward(features)

    loss = binary_cross_entropy(targets, probabilities)
    accuracy_val = accuracy(targets, probabilities)
    conf_matrix = confusion_matrix(targets, probabilities)

    print(f"evaluated samples: {len(features)}")
    print(f"binary cross-entropy: {loss:.6f}")
    print(f"accuracy: {accuracy_val:.4f}")
    print_conf_matrix(conf_matrix)
    print_classification_metrics(conf_matrix)

if __name__ == "__main__":
    main()