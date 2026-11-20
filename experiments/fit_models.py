import argparse
from pathlib import Path

from experiments.comparison_plot import (
    save_comparison_plot,
)
from experiments.history_io import (
    load_history,
    save_history,
)
from mlp.data import load_dataset
from mlp.model import MLP
from mlp.preprocessing import (
    dataframe_to_arrays,
    fit_standard_params,
    one_hot_encoding,
    standard_scaler,
)
from mlp.train_utils import (
    PreparedDataset,
    TrainingConfig,
    train_model,
)


EXPERIMENTS = [
    {
        "name": "sigmoid_12_12",
        "layers": [12, 12],
        "activation": "sigmoid",
    },
    {
        "name": "tanh_8_8",
        "layers": [8, 8],
        "activation": "tanh",
    },
    {
        "name": "relu_12_12",
        "layers": [12, 12],
        "activation": "relu",
    },
]

EPOCHS = 300
LEARNING_RATE = 0.01
BATCH_SIZE = 4
SEED = 42


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train and compare several MLP models."
    )

    parser.add_argument("training_data")
    parser.add_argument("validation_data")

    return parser.parse_args()


def main():
    args = parse_arguments()

    train_df = load_dataset(args.training_data)
    val_df = load_dataset(args.validation_data)

    train_features, train_labels = (
        dataframe_to_arrays(train_df)
    )

    val_features, val_labels = (
        dataframe_to_arrays(val_df)
    )

    mean, std = fit_standard_params(train_features)

    train_features = standard_scaler(
        train_features,
        mean,
        std,
    )

    val_features = standard_scaler(
        val_features,
        mean,
        std,
    )

    train_data = PreparedDataset(
        features=train_features,
        targets=one_hot_encoding(train_labels),
    )

    val_data = PreparedDataset(
        features=val_features,
        targets=one_hot_encoding(val_labels),
    )

    output_dir = Path("artifacts/comparisons")
    history_files = []

    for experiment in EXPERIMENTS:
        print()
        print(f"training model: {experiment['name']}")

        config = TrainingConfig(
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            seed=SEED,
            early_stopping=True,
        )

        layer_sizes = [
            train_features.shape[1],
            *experiment["layers"],
            2,
        ]

        model = MLP(
            layer_sizes=layer_sizes,
            activation=experiment["activation"],
            seed=SEED,
        )

        history = train_model(
            model,
            train_data,
            val_data,
            config,
        )

        history_path = (
            output_dir
            / f"{experiment['name']}.json"
        )

        save_history(
            history_path,
            experiment["name"],
            model,
            config,
            history,
        )

        history_files.append(history_path)

    histories = [
        load_history(history_path)
        for history_path in history_files
    ]

    save_comparison_plot(
        histories,
        output_dir / "comparison.png",
    )


if __name__ == "__main__":
    main()