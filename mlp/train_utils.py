from dataclasses import dataclass
import numpy as np

EARLY_STOP_PATIENCE = 10
EARLY_STOP_EPS = 0.0005

@dataclass
class PreparedDataset:
    features: np.ndarray
    targets: np.ndarray

@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 0.01
    batch_size: int = 8
    seed: int = 42
    early_stopping: bool = False

def binary_cross_entropy(targets, predictions):
    eps = 1e-15
    m_targets = targets[:, 1]
    m_prob = predictions[:, 1]

    m_prob = np.clip(m_prob, eps, 1.0 - eps)

    loss = -np.mean(m_targets * np.log(m_prob) + (1.0 - m_targets) * np.log(1.0-m_prob))
    return loss

def accuracy(targets, predictions):
    true_classes = np.argmax(targets, axis = 1)
    predicted_classes = np.argmax(predictions, axis = 1)
    return np.mean(true_classes == predicted_classes)

def confusion_matrix(targets, predictions):
    true_classes = np.argmax(targets, axis=1)
    predicted_classes = np.argmax(predictions, axis=1)

    tn = np.sum((true_classes == 0) & (predicted_classes == 0))
    fp = np.sum((true_classes == 0) & (predicted_classes == 1))
    tp = np.sum((true_classes == 1) & (predicted_classes == 1))
    fn = np.sum((true_classes == 1) & (predicted_classes == 0))
    return np.array([[tn, fp],[fn, tp]])

def print_conf_matrix(matrix):
    print("confusion matrix \n (rows=actual, columns=predicted)")
    print("              B     M")
    print(f"actual B   {matrix[0, 0]:5d} " f"{matrix[0, 1]:5d}")
    print(f"actual M   {matrix[1, 0]:5d} " f"{matrix[1, 1]:5d}")

def classification_metrics(matrix):
    tn = matrix[0,0]
    fp = matrix[0,1]
    fn = matrix[1,0]
    tp = matrix[1,1]

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0.0

    if precision + recall > 0:
        f1_score = (2 * precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    return precision, recall, specificity, f1_score

def print_classification_metrics(matrix):
    pr, rec, spe, f1 = classification_metrics(matrix)

    print("metrics for positive class M")
    print(f"precision:   {pr:.4f}")
    print(f"recall:      {rec:.4f}")
    print(f"specificity: {spe:.4f}")
    print(f"F1 score:    {f1:.4f}")

def train_one_epoch(model, data, config, rand_gen):
    num_of_samples = data.features.shape[0]
    shuffled_idxs = rand_gen.permutation(num_of_samples)
    for batch_start in range (0, num_of_samples, config.batch_size):
        batch_end = batch_start + config.batch_size
        batch_idxs = shuffled_idxs[batch_start:batch_end]
        batch_features = data.features[batch_idxs]
        batch_targets = data.targets[batch_idxs]

        model.forward(batch_features)
        model.backward(batch_targets)
        model.update_parameters(config.learning_rate)

def train_model(model, train_data, val_data, config):
    rand_gen = np.random.default_rng(config.seed)
    history = {"train_loss": [],
               "val_loss": [],
               "train_accuracy": [],
               "val_accuracy": []
               }
    best_val_loss = float("inf")
    best_epoch = 0
    epoch_without_improvement = 0

    
    for epoch in range(config.epochs):
        train_one_epoch(model, train_data, config, rand_gen)
        train_predictions = model.forward(train_data.features)
        train_loss = binary_cross_entropy(train_data.targets, train_predictions)
        train_accuracy = accuracy(train_data.targets, train_predictions)

        val_predictions = model.forward(val_data.features)
        val_loss = binary_cross_entropy(val_data.targets, val_predictions)
        val_accuracy = accuracy(val_data.targets, val_predictions)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"epoch {epoch + 1:03d}/{config.epochs:03d} - "
            f"loss: {train_loss:.4f} - "
            f"accuracy: {train_accuracy:.4f} - "
            f"val_loss: {val_loss:.4f} - "
            f"val_accuracy: {val_accuracy:.4f}"
        )

        if config.early_stopping:
            improvement = best_val_loss-val_loss
            if improvement > EARLY_STOP_EPS:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                epoch_without_improvement = 0
                best_weights = []
                best_biases = []
                for weights in model.weights:
                    best_weights.append(weights.copy())
                for biases in model.biases:
                    best_biases.append(biases.copy())
            else:
                epoch_without_improvement +=1
            if epoch_without_improvement >= EARLY_STOP_PATIENCE:
                print("early stopping was triggered")
                break
    if config.early_stopping:
        model.weights = best_weights
        model.biases = best_biases
        print(f"restored model from epoch {best_epoch}")
    return history
