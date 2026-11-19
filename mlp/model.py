import numpy as np
from mlp.activations import (
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    softmax,
    tanh,
    tanh_derivative
)

class MLP:
    def __init__(self, layer_sizes, activation="sigmoid", seed=42):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.weights = []
        self.biases = []

        self.init_parameters(seed)

    def init_parameters(self, seed):
        random_generator = np.random.default_rng(seed)
        for layer_idx in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[layer_idx]
            output_size = self.layer_sizes[layer_idx + 1]

            is_output_layer = (layer_idx == len(self.layer_sizes) - 2)
            if self.activation == "relu" and not is_output_layer:
                scale = np.sqrt(2.0 / input_size)
            else:
                scale = np.sqrt(2.0 / (input_size + output_size))
            weight_matrix = random_generator.normal(loc=0.0, scale=scale, size = (input_size, output_size))
            bias_vector = np.zeros((1, output_size))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def hidden_activation(self, values):
        if self.activation == "sigmoid":
            return sigmoid(values)
        if self.activation == "tanh":
            return tanh(values)
        if self.activation == "relu":
            return relu(values)
        raise ValueError(f"Unknown activation function: {self.activation}")

    def hidden_activation_derivative(self, values):
        if self.activation == "sigmoid":
            return sigmoid_derivative(values)
        if self.activation == "tanh":
            return tanh_derivative(values)
        if self.activation == "relu":
            return relu_derivative(values)
        raise ValueError(f"Unknown activation function: {self.activation}")

    def forward (self, features):
        self.activations = [features]
        self.z_values = []

        current_activation = features
        last_layer_idx = len(self.weights) - 1

        for layer_idx in range(len(self.weights)):
            z = np.dot(current_activation, self.weights[layer_idx]) + self.biases[layer_idx]
            self.z_values.append(z)

            if layer_idx == last_layer_idx:
                current_activation = softmax(z)
            else:
                current_activation = self.hidden_activation(z)
            self.activations.append(current_activation)
        return current_activation

    def backward(self, targets):
        batch_size = targets.shape[0]
        number_of_wlayers = len(self.weights)

        self.weight_gradients = [None] * number_of_wlayers
        self.bias_gradients = [None] * number_of_wlayers

        predictions = self.activations[-1]
        d_z = (predictions - targets) / batch_size

        for layer_idx in range(number_of_wlayers - 1, -1, -1):
            previous_activation = self.activations[layer_idx]

            self.weight_gradients[layer_idx] = np.dot(previous_activation.T, d_z)

            self.bias_gradients[layer_idx] = np.sum(d_z, axis=0, keepdims=True)

            if layer_idx > 0:
                d_activation = np.dot(d_z, self.weights[layer_idx].T)

                activation_derivative = (self.hidden_activation_derivative(self.z_values[layer_idx - 1]))

                d_z = (d_activation * activation_derivative)
        return self.weight_gradients, self.bias_gradients

    def update_parameters(self, learning_rate):
        for idx in range(len(self.weights)):
            self.weights[idx] -= learning_rate * self.weight_gradients[idx]
            self.biases[idx] -= learning_rate * self.bias_gradients[idx]
