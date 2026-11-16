import numpy as np

def sigmoid(values):
    return 1.0/(1.0 + np.exp(-values))

def sigmoid_derivative(values):
    sigmoid_val = sigmoid(values)
    return sigmoid_val * (1.0 - sigmoid_val)

def tanh(values):
    return np.tanh(values)

def tanh_derivative(values):
    tanh_val = tanh(values)
    return 1.0 - tanh_val * tanh_val

def relu(values):
    return np.maximum(0.0, values)

def relu_derivative(values):
    return (values > 0).astype(float)

def softmax(values):
    shifted_val = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted_val)
    return exp_values / exp_values.sum(axis=1, keepdims=True)

