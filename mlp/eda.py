"""Exploratory data analysis for the breast-cancer dataset."""

import argparse
from pathlib import Path

import pandas as pd

from mlp.data import FEATURE_COLUMNS, load_dataset


DEFAULT_DATASET_PATH = Path("data/data.csv")
PREVIEW_COLUMNS = ["id", "diagnosis", *FEATURE_COLUMNS[:5]]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Explore the Wisconsin breast-cancer dataset."
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=(
            "Path to the dataset CSV file "
            f"(default: {DEFAULT_DATASET_PATH})"
        ),
    )

    return parser.parse_args()


def build_class_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return class counts and percentages."""

    counts = (
        dataframe["diagnosis"]
        .value_counts()
        .reindex(["B", "M"], fill_value=0)
    )

    percentages = counts / len(dataframe) * 100

    summary = pd.DataFrame(
        {
            "count": counts,
            "percentage": percentages.round(2),
        }
    )

    summary.index.name = "diagnosis"

    return summary


def build_feature_statistics(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return basic descriptive statistics for numerical features."""

    statistics = (
        dataframe[FEATURE_COLUMNS]
        .agg(["min", "max", "mean", "std"])
        .transpose()
    )

    statistics.index.name = "feature"

    return statistics


def print_overview(dataframe: pd.DataFrame) -> None:
    """Print the general dataset structure and quality checks."""

    print("Dataset overview")
    print("----------------")
    print(f"Rows: {dataframe.shape[0]}")
    print(f"Columns: {dataframe.shape[1]}")
    print(f"Numerical features: {len(FEATURE_COLUMNS)}")
    print(f"Missing values: {int(dataframe.isna().sum().sum())}")
    print(f"Duplicate IDs: {int(dataframe['id'].duplicated().sum())}")

    print("\nFirst five rows")
    print("---------------")
    print(dataframe[PREVIEW_COLUMNS].head().to_string(index=False))


def print_class_distribution(dataframe: pd.DataFrame) -> None:
    """Print the target-class distribution."""

    print("\nClass distribution")
    print("------------------")
    print(build_class_summary(dataframe).to_string())


def print_feature_statistics(dataframe: pd.DataFrame) -> None:
    """Print feature minimums, maximums, means, and deviations."""

    statistics = build_feature_statistics(dataframe)

    print("\nFeature statistics")
    print("------------------")
    print(statistics.round(6).to_string())

    smallest_mean_feature = statistics["mean"].idxmin()
    largest_mean_feature = statistics["mean"].idxmax()

    print("\nFeature-scale examples")
    print("----------------------")
    print(
        f"Smallest mean: {smallest_mean_feature} = "
        f"{statistics.loc[smallest_mean_feature, 'mean']:.6f}"
    )
    print(
        f"Largest mean: {largest_mean_feature} = "
        f"{statistics.loc[largest_mean_feature, 'mean']:.6f}"
    )


def main() -> None:
    """Run the exploratory data analysis."""

    arguments = parse_arguments()
    dataframe = load_dataset(arguments.dataset)

    print(f"Dataset: {arguments.dataset}\n")

    print_overview(dataframe)
    print_class_distribution(dataframe)
    print_feature_statistics(dataframe)


if __name__ == "__main__":
    main()