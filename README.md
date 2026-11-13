# Multilayer Perceptron - Python rewrite

This repository contains a from-scratch implementation of a multilayer perceptron
for binary breast-cancer classification. The new implementation will be written in
Python for the 42 Machine Learning project.

The previous C++ implementation is preserved in [`cpp_version/`](cpp_version/).

## Project source

- Subject: `en.subject.pdf`, version 5.1.
- Dataset: `data/data.csv`.
- Dataset shape: 569 rows and 32 columns, without a header.
- Column 0: sample identifier.
- Column 1: diagnosis (`M` = malignant, `B` = benign).
- Columns 2-31: 30 numerical features.
- Class distribution: 212 malignant and 357 benign samples.

## Rules from the subject

- The neural network and its learning algorithms must be implemented from scratch.
- Libraries may be used for linear algebra and plotting learning curves.
- Neural-network and machine-learning libraries must not implement the model for us.
- The default network must contain at least two hidden layers.
- The output layer must use softmax and produce a probability distribution.
- The raw dataset must be preprocessed before training.
- The dataset must be split into training and validation sets by our program.
- Training must use feedforward, backpropagation, and gradient descent.
- Training and validation metrics must be printed at every epoch.
- Loss and accuracy learning curves must be displayed or saved after training.
- The trained model must be saved with its topology and learned parameters.
- Prediction must load the saved model and evaluate a supplied dataset with binary
  cross-entropy.
- The repository must expose three phases: split, train, and predict. They may be
  separate programs or modes of one program.
- Only files committed to the Git repository are evaluated.
- Bonus work is evaluated only when the mandatory part is complete and reliable.

## Planned mandatory interface

We will use three small command-line programs because they are easy to demonstrate
and easy to explain during the defense:

```text
python3 split.py data/data.csv
python3 train.py data/data_training.csv data/data_validation.csv
python3 predict.py artifacts/model.npz data/data_validation.csv
```

The exact options will be finalized while implementing each phase. At minimum, the
training interface should support:

```text
--layers 24 24
--epochs 100
--learning-rate 0.01
--batch-size 8
--seed 42
```

## Planned project structure

```text
.
├── README.md
├── en.subject.pdf
├── evaluation.py
├── data/
│   └── data.csv
├── mlp/
│   ├── __init__.py
│   ├── activations.py
│   ├── data.py
│   ├── layers.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── serialization.py
│   └── training.py
├── split.py
├── train.py
├── predict.py
├── tests/
├── artifacts/            # generated models, scaler, history, and plots
└── cpp_version/          # archived previous implementation
```

The structure may change slightly if a simpler organization becomes clearer during
implementation.

## Mathematical reference

For a dense layer `l`:

```text
Z[l] = A[l-1] @ W[l] + b[l]
A[l] = activation(Z[l])
```

For a numerically stable softmax:

```text
shifted_logits = Z - max(Z, axis=1)
softmax(Z_i) = exp(shifted_logits_i) / sum(exp(shifted_logits))
```

For the malignant-class probability `p` and target `y`:

```text
BCE = -(1/N) * sum(y * log(p) + (1-y) * log(1-p))
```

Gradient descent updates every trainable parameter with:

```text
parameter = parameter - learning_rate * gradient
```

The derivation of feedforward and backpropagation will be documented alongside the
implementation so every equation can be explained during the defense.

# Progress checklist

## 0. Repository preparation

- [x] Read and inspect all 13 pages of subject version 5.1.
- [x] Preserve the previous C++ implementation in `cpp_version/`.
- [x] Confirm the raw dataset format, size, labels, and absence of empty cells.
- [x] Update `.gitignore` for Python caches, virtual environments, generated models,
      histories, split datasets, and plots.
- [x] Add a minimal dependency file containing only allowed libraries.
- [x] Create the agreed Python package and test directories.
- [x] Confirm that the project runs with a clean Python environment.

Completion criterion: the repository has a clean Python-oriented skeleton without
mixing generated files with source code.

## 1. Dataset exploration and understanding

- [ ] Document the meaning of the identifier, diagnosis, and 30 feature columns.
- [ ] Verify that all feature values can be parsed as finite numbers.
- [ ] Compute basic statistics for every feature: minimum, maximum, mean, and standard
      deviation.
- [ ] Inspect class imbalance (`M` versus `B`).
- [ ] Identify constant or near-constant features, if any.
- [ ] Produce simple exploratory plots without using a machine-learning library.
- [ ] Document why the identifier column must not be used as a feature.
- [ ] Decide and document the label encoding (`B = 0`, `M = 1`).

Completion criterion: we understand exactly what enters the network and can explain
every preprocessing decision.

## 2. Reproducible train/validation split

- [ ] Implement CSV loading with clear validation and useful error messages.
- [ ] Implement a deterministic shuffle controlled by `--seed`.
- [ ] Implement a stratified split so both sets preserve the class distribution.
- [ ] Support a configurable validation ratio with a documented default.
- [ ] Write separate training and validation CSV files.
- [ ] Prevent accidental overwriting unless explicitly requested.
- [ ] Print row counts and class counts for both output files.
- [ ] Test that the same seed produces identical files.
- [ ] Test that different seeds change the split.
- [ ] Test that every original row occurs exactly once across the two outputs.

Completion criterion: `split.py` reliably creates repeatable, non-overlapping training
and validation datasets.

## 3. Preprocessing without data leakage

- [ ] Remove the identifier column from model inputs.
- [ ] Encode labels consistently.
- [ ] Fit feature normalization using the training set only.
- [ ] Standardize each feature with `(x - mean) / std`.
- [ ] Handle a zero standard deviation safely.
- [ ] Apply training statistics to the validation and prediction datasets.
- [ ] Save preprocessing statistics together with the trained model.
- [ ] Validate input column count and reject malformed rows.
- [ ] Test preprocessing shapes and numerical values on small known examples.

Completion criterion: preprocessing is deterministic, reusable for prediction, and
does not leak validation information into training.

## 4. Neural-network building blocks

- [ ] Implement a dense layer with weights and biases.
- [ ] Implement a clear parameter-shape convention and document it.
- [ ] Implement sigmoid for hidden layers.
- [ ] Implement numerically stable softmax for the output layer.
- [ ] Implement binary cross-entropy with probability clipping for numerical safety.
- [ ] Implement reproducible weight initialization with a seed.
- [ ] Use an initialization appropriate for sigmoid, such as Xavier initialization.
- [ ] Support a configurable list of hidden-layer sizes.
- [ ] Enforce at least two hidden layers in the default configuration.
- [ ] Implement feedforward for a complete batch.
- [ ] Cache only the intermediate values required by backpropagation.
- [ ] Test that softmax rows sum to 1.
- [ ] Test all output and gradient shapes.
- [ ] Test that loss remains finite for probabilities close to 0 or 1.

Completion criterion: a network with input size 30, at least two hidden layers, and a
two-neuron softmax output can perform a stable forward pass.

## 5. Backpropagation and gradient descent

- [ ] Derive the output-layer gradient on paper/in documentation.
- [ ] Derive the dense-layer gradients for weights, biases, and previous activations.
- [ ] Implement backpropagation without an automatic-differentiation library.
- [ ] Average gradients consistently over the batch.
- [ ] Implement mini-batch gradient descent.
- [ ] Shuffle training samples reproducibly at every epoch.
- [ ] Update every weight and bias using the configured learning rate.
- [ ] Add a finite-difference gradient check on a tiny network.
- [ ] Confirm that the analytical and numerical gradients agree within tolerance.
- [ ] Confirm that training can overfit a very small sample.

Completion criterion: gradients are mathematically verified, loss decreases on a
controlled example, and no neural-network framework is involved.

## 6. Training program

- [ ] Implement argument parsing with helpful descriptions and validation.
- [ ] Load the training and validation files explicitly.
- [ ] Build the default architecture with at least two hidden layers.
- [ ] Support configurable layers, epochs, learning rate, batch size, and seed.
- [ ] Print training and validation loss at every epoch.
- [ ] Print training and validation accuracy at every epoch.
- [ ] Keep epoch numbering and metric formatting easy to read.
- [ ] Record the metrics required to generate learning curves.
- [ ] Save the network topology, weights, biases, label mapping, and scaler statistics.
- [ ] Save enough metadata to reproduce the training configuration.
- [ ] Generate separate loss and accuracy graphs for training and validation.
- [ ] Handle invalid paths and options without a traceback intended for developers.
- [ ] Test deterministic training with a fixed seed.

Completion criterion: `train.py` trains the model end to end, prints both datasets'
metrics per epoch, creates two required graphs, and saves a loadable model.

## 7. Prediction and mandatory evaluation

- [ ] Load and validate the saved model file.
- [ ] Restore the exact topology and preprocessing statistics.
- [ ] Load and preprocess a supplied evaluation CSV.
- [ ] Produce a probability distribution with softmax for every sample.
- [ ] Convert probabilities to predicted `M` or `B` labels.
- [ ] Compute binary cross-entropy using the formula required by the subject.
- [ ] Compute accuracy to make the result easy to interpret.
- [ ] Print the number of evaluated samples and the final metrics clearly.
- [ ] Define and document the decision threshold.
- [ ] Test model save/load round trips.
- [ ] Test that predictions before and after loading are identical.
- [ ] Reject incompatible model or dataset files with clear messages.

Completion criterion: `predict.py` loads a model trained in a separate process and
evaluates unknown examples with the required binary cross-entropy.

## 8. Mandatory quality gate

- [ ] Run the complete workflow from the raw dataset: split, train, predict.
- [ ] Run the workflow from a clean clone or clean environment.
- [ ] Confirm that no generated artifact is required from outside the repository.
- [ ] Confirm that no forbidden ML/NN library is imported.
- [ ] Confirm that the default model has at least two hidden layers.
- [ ] Confirm that the output layer is softmax.
- [ ] Confirm that training uses our own backpropagation and gradient descent.
- [ ] Confirm that validation data is not used to fit the scaler or update weights.
- [ ] Confirm that every epoch displays training and validation metrics.
- [ ] Confirm that both mandatory learning curves are created.
- [ ] Confirm that the saved model contains topology and learned parameters.
- [ ] Confirm that prediction reports binary cross-entropy.
- [ ] Add automated tests for the critical mathematical and data-processing behavior.
- [ ] Add copy-paste-ready usage examples to this README.
- [ ] Review all file and directory names before submission.

Completion criterion: every mandatory requirement works without malfunctioning. Only
after this gate is fully checked do we start counting bonus work as complete.

## 9. Bonus implementation

The goal is to implement all bonuses suggested by the subject, but only after the
mandatory quality gate passes.

### 9.1 Advanced optimizer

- [ ] Define a small optimizer interface separate from the network layers.
- [ ] Keep plain stochastic/mini-batch gradient descent as the mandatory baseline.
- [ ] Implement Adam from its equations, including first and second moments.
- [ ] Implement bias correction and numerical epsilon handling.
- [ ] Allow optimizer selection from the command line.
- [ ] Test Adam updates on known gradients.
- [ ] Compare Adam with baseline gradient descent using the same split and seed.

### 9.2 Persistent training history

- [ ] Save per-epoch metrics to a documented CSV or JSON file.
- [ ] Include the training configuration and random seed with the history.
- [ ] Make history writing robust to interrupted or failed runs.
- [ ] Add a script or command that reloads history without retraining.

### 9.3 Early stopping

- [ ] Monitor validation loss.
- [ ] Support configurable patience and minimum improvement.
- [ ] Remember the best model parameters rather than only the last epoch.
- [ ] Restore the best parameters when training stops.
- [ ] Print the stopping reason and best epoch.
- [ ] Test early stopping with a controlled metric sequence.

### 9.4 Multiple evaluation metrics

- [ ] Implement a confusion matrix from predictions and labels.
- [ ] Implement precision, recall, and F1 score from scratch.
- [ ] Define which class is considered positive (`M`).
- [ ] Handle divisions by zero explicitly.
- [ ] Display mandatory loss and accuracy plus the bonus metrics.
- [ ] Add tests using small, hand-calculated examples.

### 9.5 Model and curve comparison

- [ ] Load histories from multiple training runs.
- [ ] Plot multiple loss curves on one graph.
- [ ] Plot multiple accuracy curves on one graph.
- [ ] Label every model and configuration clearly.
- [ ] Compare models on the same split for a fair result.
- [ ] Document conclusions instead of selecting a model only by training accuracy.

Completion criterion: bonus features are configurable, tested, documented, and do
not complicate or break the mandatory path.

## 10. Defense preparation

- [ ] Explain a perceptron, weight, bias, activation, dense layer, and MLP.
- [ ] Explain every tensor/matrix shape in the default network.
- [ ] Explain feedforward using the implementation and equations.
- [ ] Explain the purpose and numerical stability of softmax.
- [ ] Explain binary cross-entropy and why lower is better.
- [ ] Explain the chain rule and backpropagation layer by layer.
- [ ] Explain gradient descent and the learning-rate tradeoff.
- [ ] Explain mini-batches, epochs, shuffling, and seeds.
- [ ] Explain Xavier initialization.
- [ ] Explain why normalization is fitted only on training data.
- [ ] Explain the distinction between training and validation metrics.
- [ ] Explain underfitting, overfitting, and how the curves reveal them.
- [ ] Explain each implemented bonus from its equations, not only its API.
- [ ] Rehearse the full split -> train -> predict demonstration.
- [ ] Prepare answers about design choices and known limitations.

Completion criterion: every major line of the learning algorithm can be explained and
defended without relying on a library's hidden implementation.

## 11. Final submission

- [ ] Run all tests successfully.
- [ ] Run all three command-line programs with the documented examples.
- [ ] Check Git status and remove accidental generated files from tracking.
- [ ] Confirm that all required source files are committed.
- [ ] Confirm that the repository contains no secrets or machine-specific paths.
- [ ] Confirm that the project can be installed and run from the README alone.
- [ ] Review the final Git diff and commit history.
- [ ] Push the final branch to the Git repository.

## Progress rule

We will work from top to bottom. A checkbox is marked complete only after the code is
implemented, tested, and understood well enough to explain during the defense.
