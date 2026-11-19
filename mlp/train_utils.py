from dataclasses import dataclass
import numpy as np

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

    return history
