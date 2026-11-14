"""Split a dataset into stratified training and validation sets."""

import argparse
from pathlib import Path

import pandas as pd

from mlp.data import load_dataset


def split_dataset(dataframe, validation_ratio, seed):
    """Split the dataset while preserving the class distribution."""

    if not 0 < validation_ratio < 1:
        raise ValueError(
            "Validation ratio must be greater than 0 and less than 1"
        )

    training_parts = []
    validation_parts = []

    for diagnosis in ["B", "M"]:
        class_data = dataframe[
            dataframe["diagnosis"] == diagnosis
        ]

        shuffled_class_data = class_data.sample(
            frac=1,
            random_state=seed,
        ).reset_index(drop=True)

        validation_size = round(
            len(shuffled_class_data) * validation_ratio
        )

        validation_part = shuffled_class_data.iloc[
            :validation_size
        ]

        training_part = shuffled_class_data.iloc[
            validation_size:
        ]

        validation_parts.append(validation_part)
        training_parts.append(training_part)

    training_data = pd.concat(
        training_parts,
        ignore_index=True,
    )

    validation_data = pd.concat(
        validation_parts,
        ignore_index=True,
    )

    training_data = training_data.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    validation_data = validation_data.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    return training_data, validation_data


def print_split_summary(name, dataframe):
    """Print the number of rows and classes in a dataset."""

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Rows: {len(dataframe)}")
    print(dataframe["diagnosis"].value_counts().to_string())


def parse_arguments():
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Split a breast-cancer dataset into "
            "training and validation sets."
        )
    )

    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the source CSV dataset",
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Validation-set ratio (default: 0.2)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    return parser.parse_args()


def main():
    """Load, split, and save the dataset."""

    arguments = parse_arguments()

    dataframe = load_dataset(arguments.dataset)
    

    training_data, validation_data = split_dataset(
        dataframe=dataframe,
        validation_ratio=arguments.validation_ratio,
        seed=arguments.seed,
    )

    training_path = arguments.dataset.with_name(
        f"{arguments.dataset.stem}_training.csv"
    )

    validation_path = arguments.dataset.with_name(
        f"{arguments.dataset.stem}_validation.csv"
    )

    training_data.to_csv(
        training_path,
        index=False,
        header=False,
    )

    validation_data.to_csv(
        validation_path,
        index=False,
        header=False,
    )

    print(f"Seed: {arguments.seed}")
    print(f"Validation ratio: {arguments.validation_ratio}")

    print_split_summary("Training dataset", training_data)
    print_split_summary("Validation dataset", validation_data)

    print("\nFiles saved:")
    print(training_path)
    print(validation_path)


if __name__ == "__main__":
    main()