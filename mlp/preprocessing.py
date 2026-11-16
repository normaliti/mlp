import pandas as pd
import numpy as np

from mlp.data import FEATURE_COLUMNS

LABEL_MAPPING = {"B": 0, "M": 1}

NUMBER_OF_CLASSES = 2

def dataframe_to_arrays(df):
    features = df[FEATURE_COLUMNS].to_numpy(dtype = float)
    labels = df["diagnosis"].map(LABEL_MAPPING).to_numpy(dtype = int)
    return features, labels

def fit_standard_params(features):
    mean = features.mean(axis = 0)
    std = features.std(axis = 0)
    std[std == 0] = 1.0
    return mean, std

def standard_scaler(features, mean, std):
    return (features - mean) / std

def one_hot_encoding(labels):
    encoded_labels = np.zeros((len(labels), NUMBER_OF_CLASSES), dtype = float)     
    row_indices = np.arange(len(labels))
    encoded_labels[row_indices, labels] = 1.0
    return encoded_labels



