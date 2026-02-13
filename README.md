# Multilayer Perceptron (MLP)

This project implements a multilayer perceptron from scratch in C++.

## Build

```
make
```

## Programs

### 1) Split
Stratified split into train/validation.

```
./split <input_csv> <train_csv> <valid_csv> [seed] [valid_ratio]
```

Example:
```
./split data.csv data_training.csv data_validation.csv 42 0.2
```

Arguments:
- `input_csv`  : source dataset (raw)
- `train_csv`  : output train dataset
- `valid_csv`  : output validation dataset
- `seed`       : optional, random seed (default 42)
- `valid_ratio`: optional, validation ratio (default 0.2)

### 2) Train
Trains the network, saves scaler, model, and training history.

```
./train <train_csv> <valid_csv> [scaler_out] [model_out] [options]
```

Default outputs:
- `models/scaler.txt`
- `models/model.txt`
- `models/history.csv`

Example:
```
./train data_training.csv data_validation.csv --epochs 50 --batch_size 8 --lr 0.1
```

Options:
- `--layers 30 24 24 2`  (default: 30 24 24 2)
- `--epochs 50`          (default: 50)
- `--batch_size 8`       (default: 8)
- `--lr 0.1`             (default: 0.01)
- `--activation sigmoid|tanh|relu` (default: sigmoid)

You can also pass explicit output paths:
```
./train data_training.csv data_validation.csv models/scaler.txt models/model.txt --epochs 50 --batch_size 8 --lr 0.1
```

### 3) Predict
Loads model + scaler and evaluates a dataset.

```
./predict <model_path> <scaler_path> <data_csv>
```

Example:
```
./predict models/model.txt models/scaler.txt data_validation.csv
```
Outputs:
- Binary cross-entropy, accuracy, precision, recall, F1
- `models/roc.csv` and `models/roc.png`

## Plots
Training history is saved to `models/history.csv`.
Build graphs with:

```
python3 scripts/plot.py
```

Outputs:
- `models/loss.png`
- `models/accuracy.png`
Predict generates:
- `models/roc.png`

You can pass multiple history files to compare curves:
```
python3 scripts/plot.py models/history_run1.csv models/history_run2.csv
```

## Project Structure

- `src/` — source files
- `include/` — headers
- `scripts/` — helper scripts (plots)
- `models/` — saved models, scaler, training history
- `data/` — data directory (optional)

## Dependencies

- C++17 compiler (for `std::filesystem`)
- Python 3 + matplotlib (for plots)

`make` will automatically install matplotlib (via `make deps`) if it is missing.
