import argparse

from mlp.data import load_dataset
from mlp.model import MLP
from mlp.preprocessing import (
    dataframe_to_arrays,
    fit_standard_params,
    one_hot_encoding,
    standard_scaler
)
from mlp.train_utils import (
    PreparedDataset,
    TrainingConfig,
    train_model
)

from mlp.model_io import save_model


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a multilayer perceptron."
    )

    parser.add_argument("training_data")
    parser.add_argument("validation_data")

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--activation",
        choices=["sigmoid", "tanh", "relu"],
        default="sigmoid",
    )
    parser.add_argument(
        "--layers",
        type = int,
        nargs="+",
        default=[24,24],
        help="sizes of hidden layers"
    )
    parser.add_argument("--model-output", 
                        default="artifacts/model.npz",
                        help="Path for saving the trained model"
                        )
    args = parser.parse_args()
    if len(args.layers) < 2:
        parser.error("At least 2 hidden layers are required")
    for l_size in args.layers:
        if l_size <= 0:
            parser.error ("Layer size must be positive")
    return args

def main():
    args = parse_arguments()

    train_df = load_dataset(args.training_data)
    val_df = load_dataset(args.validation_data)

    train_features, train_labels = dataframe_to_arrays(train_df)
    val_features, val_labels = dataframe_to_arrays(val_df)

    mean, std = fit_standard_params(train_features)
    train_features = standard_scaler(train_features, mean, std)
    val_features = standard_scaler(val_features, mean, std)

    train_targets = one_hot_encoding(train_labels)
    val_targets = one_hot_encoding(val_labels)

    train_data = PreparedDataset(features=train_features, targets=train_targets)
    val_data = PreparedDataset(features=val_features, targets=val_targets)

    config = TrainingConfig(epochs=args.epochs,
                            learning_rate=args.learning_rate,
                            batch_size=args.batch_size,
                            seed = args.seed)
    layer_sizes = [train_features.shape[1], *args.layers, 2]
    model = MLP(layer_sizes=layer_sizes,
                activation=args.activation,
                seed = args.seed)
    print(f"train shape: {train_features.shape}")
    print(f"validation shape: {val_features.shape}")

    history = train_model(model, train_data, val_data, config)
    save_model(args.model_output, model, mean, std)

if __name__ == "__main__":
    main()
    