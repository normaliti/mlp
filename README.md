# Multilayer Perceptron

From-scratch implementation of a multilayer perceptron for binary breast-cancer
classification (`B` = benign, `M` = malignant).

The neural network, feedforward pass, backpropagation, gradient descent, activation
functions, loss, and metrics are implemented without a machine-learning framework.
NumPy is used for numerical operations, Pandas for CSV loading, and Matplotlib for
learning curves.

## Requirements

- Python 3.9 or newer
- NumPy
- Pandas
- Matplotlib

Run every command below from the repository root.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick verification

### 1. Split the dataset

```bash
python -m mlp.split data/data.csv \
  --validation-ratio 0.2 \
  --seed 42
```

This creates:

```text
data/data_training.csv
data/data_validation.csv
```

### 2. Train and save one model

```bash
python -m mlp.train \
  data/data_training.csv \
  data/data_validation.csv \
  --layers 12 12 \
  --activation sigmoid \
  --learning-rate 0.01 \
  --batch-size 4 \
  --epochs 300 \
  --seed 42 \
  --early-stopping \
  --model-output artifacts/model.npz \
  --plots-dir artifacts/plots
```

The training program prints training and validation loss and accuracy after every
epoch. It also saves:

```text
artifacts/model.npz
artifacts/plots/loss.png
artifacts/plots/accuracy.png
```

### 3. Load the model and run prediction

```bash
python -m mlp.predict \
  artifacts/model.npz \
  data/data_validation.csv
```

The prediction program prints:

- binary cross-entropy;
- accuracy;
- confusion matrix;
- precision, recall, specificity, and F1 score for class `M`.


## Bonus: compare several models

The experiment runner trains several architectures on the same split, saves each
training history as JSON, and creates one comparison figure with loss and accuracy
curves.

For the quick-verification split:

```bash
python -m experiments.fit_models \
  data/data_training.csv \
  data/data_validation.csv
```

Results are written to:

```text
artifacts/comparisons/
```

## Command help

```bash
python -m mlp.split --help
python -m mlp.train --help
python -m mlp.predict --help
```

Supported hidden-layer activations are `sigmoid`, `tanh`, and `relu`. The number and
size of hidden layers are configured with `--layers`; at least two hidden layers are
required.
