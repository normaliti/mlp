import numpy as np

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

