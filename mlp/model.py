import numpy as np


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
