#include "mlp.hpp"
#include "dataset.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

namespace mlp {

namespace {

bool is_valid_dataset_for_model(const dataset::Dataset& data,
                                int input_size,
                                int num_classes) {
    if (data.x.empty() || data.y.size() != data.x.size()) {
        return false;
    }

    for (size_t i = 0; i < data.x.size(); ++i) {
        if (data.x[i].size() != static_cast<size_t>(input_size)) {
            return false;
        }
        const int y = data.y[i];
        if (y < 0 || y >= num_classes) {
            return false;
        }
    }
    return true;
}

} // namespace

MLP::Metrics MLP::evaluate(const dataset::Dataset& data) const {
    Metrics m;
    m.loss = 0.0;
    m.acc = 0.0;

    if (data.x.empty()) {
        return m;
    }

    int correct = 0;
    for (size_t i = 0; i < data.x.size(); ++i) {
        const Vec out = forward(data.x[i], nullptr);
        if (out.size() < 2) {
            continue;
        }

        const double p = out[1];
        m.loss += binary_cross_entropy(p, data.y[i]);

        const int pred = (out[0] > out[1]) ? 0 : 1;
        if (pred == data.y[i]) {
            ++correct;
        }
    }

    m.loss /= static_cast<double>(data.x.size());
    m.acc = static_cast<double>(correct) / static_cast<double>(data.x.size());
    return m;
}

void MLP::train(const MLPConfig& cfg, const dataset::Dataset& train, const dataset::Dataset& valid) {
    if (cfg.layers.size() < 2 || cfg.batch_size <= 0 || cfg.epochs <= 0 || cfg.learning_rate <= 0.0) {
        std::cout << "Error: invalid training configuration.\n";
        return;
    }

    const int input_size = cfg.layers.front();
    const int output_size = cfg.layers.back();
    if (output_size <= 0) {
        std::cout << "Error: output layer size must be positive.\n";
        return;
    }

    if (!is_valid_dataset_for_model(train, input_size, output_size)) {
        std::cout << "Error: training dataset shape/labels are incompatible with model config.\n";
        return;
    }
    if (!is_valid_dataset_for_model(valid, input_size, output_size)) {
        std::cout << "Error: validation dataset shape/labels are incompatible with model config.\n";
        return;
    }

    std::cout << "Layers: ";
    for (size_t i = 0; i < cfg.layers.size(); ++i) {
        std::cout << cfg.layers[i] << (i + 1 < cfg.layers.size() ? " " : "");
    }
    std::cout << "\n";
    std::cout << "Epochs: " << cfg.epochs
              << ", batch_size: " << cfg.batch_size
              << ", lr: " << cfg.learning_rate
              << ", seed: " << cfg.seed << "\n";

    activation_ = cfg.activation;
    init_weights(cfg.layers, cfg.seed);

    std::ofstream history_file(cfg.history_path.c_str());
    if (history_file) {
        history_file << "epoch,loss,acc,val_loss,val_acc\n";
    } else {
        std::cout << "Warning: could not open history file: " << cfg.history_path << "\n";
    }

    const size_t n_train = train.x.size();
    std::vector<size_t> indices(n_train);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(cfg.seed);

    for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        for (size_t start = 0; start < n_train; start += static_cast<size_t>(cfg.batch_size)) {
            const size_t end = std::min(start + static_cast<size_t>(cfg.batch_size), n_train);
            const size_t batch_count = end - start;
            if (batch_count == 0) {
                continue;
            }

            std::vector<Layer> grads(layers_.size());
            for (size_t l = 0; l < layers_.size(); ++l) {
                grads[l].W.assign(layers_[l].W.size(), Vec(layers_[l].W[0].size(), 0.0));
                grads[l].b.assign(layers_[l].b.size(), 0.0);
            }

            for (size_t bi = start; bi < end; ++bi) {
                const size_t idx = indices[bi];
                ForwardCache cache;
                forward(train.x[idx], &cache);
                backprop(cache.A, cache.Z, train.y[idx], grads);
            }

            const double lr = cfg.learning_rate / static_cast<double>(batch_count);
            apply_grads(grads, lr);
        }

        const Metrics train_metrics = evaluate(train);
        const Metrics valid_metrics = evaluate(valid);

        std::cout << "epoch " << epoch << "/" << cfg.epochs
                  << " - loss: " << train_metrics.loss
                  << " - acc: " << train_metrics.acc
                  << " - val_loss: " << valid_metrics.loss
                  << " - val_acc: " << valid_metrics.acc << "\n";

        if (history_file) {
            history_file << epoch << "," << train_metrics.loss << "," << train_metrics.acc << ","
                         << valid_metrics.loss << "," << valid_metrics.acc << "\n";
        }
    }
}

} // namespace mlp
