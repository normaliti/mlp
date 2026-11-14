"""Dataset loading and validation utilities."""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]

FEATURE_BASE_NAMES = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
]

FEATURE_GROUPS = ["mean", "se", "worst"]

FEATURE_COLUMNS = [
    f"{feature}_{group}"
    for group in FEATURE_GROUPS
    for feature in FEATURE_BASE_NAMES
]

COLUMN_NAMES = ["id", "diagnosis", *FEATURE_COLUMNS]

VALID_DIAGNOSES = {"M", "B"}


def load_dataset(path: PathLike) -> pd.DataFrame:
    """Load and validate a Wisconsin breast-cancer dataset CSV file."""

    dataset_path = Path(path)

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    try:
        dataframe = pd.read_csv(dataset_path, header=None, dtype=str)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Dataset is empty: {dataset_path}") from error

    if dataframe.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            "Invalid number of columns: "
            f"expected {len(COLUMN_NAMES)}, got {dataframe.shape[1]}"
        )

    dataframe.columns = COLUMN_NAMES
    dataframe = dataframe.apply(lambda column: column.str.strip())

    missing_mask = dataframe.isna() | dataframe.eq("")

    if missing_mask.any().any():
        row_numbers = (
            np.flatnonzero(missing_mask.any(axis=1).to_numpy()) + 1
        ).tolist()

        raise ValueError(
            f"Dataset contains missing values in rows: {row_numbers}"
        )

    if dataframe["id"].duplicated().any():
        duplicate_ids = (
            dataframe.loc[dataframe["id"].duplicated(), "id"]
            .unique()
            .tolist()
        )

        raise ValueError(f"Dataset contains duplicate IDs: {duplicate_ids}")

    diagnoses = set(dataframe["diagnosis"].unique())
    invalid_diagnoses = diagnoses - VALID_DIAGNOSES

    if invalid_diagnoses:
        raise ValueError(
            f"Dataset contains invalid diagnoses: {sorted(invalid_diagnoses)}"
        )

    numeric_features = dataframe[FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if numeric_features.isna().any().any():
        raise ValueError("Dataset contains non-numeric feature values")

    feature_values = numeric_features.to_numpy(dtype=float)

    if not np.isfinite(feature_values).all():
        raise ValueError("Dataset contains infinite feature values")

    return pd.concat(
        [
            dataframe[["id", "diagnosis"]],
            numeric_features,
        ],
        axis=1,
    )