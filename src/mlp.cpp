#include "mlp.hpp"
#include "dataset.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

namespace mlp {

MLP::MLP() {}

static double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

static std::vector<double> softmax(const std::vector<double>& z) {
    std::vector<double> out(z.size(), 0.0);
    if (z.empty()) {
        return out;
    }
    double maxv = z[0];
    for (size_t i = 1; i < z.size(); ++i) {
        if (z[i] > maxv) maxv = z[i];
    }
    double sum = 0.0;
    for (size_t i = 0; i < z.size(); ++i) {
        out[i] = std::exp(z[i] - maxv);
        sum += out[i];
    }
    if (sum == 0.0) {
        return out;
    }
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] /= sum;
    }
    return out;
}

static double cross_entropy(const std::vector<double>& probs, int y) {
    const double eps = 1e-12;
    double p = probs[y];
    if (p < eps) p = eps;
    return -std::log(p);
}

void MLP::init_weights(const std::vector<int>& sizes, unsigned int seed) {
    layers_.clear();
    sizes_ = sizes;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    for (size_t l = 0; l + 1 < sizes.size(); ++l) {
        int in = sizes[l];
        int out = sizes[l + 1];
        Layer layer;
        layer.W.assign(out, std::vector<double>(in, 0.0));
        layer.b.assign(out, 0.0);
        for (int i = 0; i < out; ++i) {
            for (int j = 0; j < in; ++j) {
                layer.W[i][j] = dist(rng);
            }
            layer.b[i] = dist(rng);
        }
        layers_.push_back(layer);
    }
}

std::vector<double> MLP::forward_one(const std::vector<double>& x) const {
    std::vector<double> a = x;
    for (size_t l = 0; l < layers_.size(); ++l) {
        const Layer& layer = layers_[l];
        std::vector<double> z(layer.W.size(), 0.0);
        for (size_t i = 0; i < layer.W.size(); ++i) {
            double sum = layer.b[i];
            for (size_t j = 0; j < layer.W[i].size(); ++j) {
                sum += layer.W[i][j] * a[j];
            }
            z[i] = sum;
        }
        if (l + 1 == layers_.size()) {
            a = softmax(z);
        } else {
            a.resize(z.size());
            for (size_t i = 0; i < z.size(); ++i) {
                a[i] = sigmoid(z[i]);
            }
        }
    }
    return a;
}

void MLP::train(const MLPConfig& cfg, const dataset::Dataset& train, const dataset::Dataset& valid) {
    std::cout << "Layers: ";
    for (size_t i = 0; i < cfg.layers.size(); ++i) {
        std::cout << cfg.layers[i] << (i + 1 < cfg.layers.size() ? " " : "");
    }
    std::cout << "\n";
    std::cout << "Epochs: " << cfg.epochs
              << ", batch_size: " << cfg.batch_size
              << ", lr: " << cfg.learning_rate << std::endl;

    init_weights(cfg.layers, 42);

    size_t n_train = train.x.size();
    std::vector<size_t> indices(n_train);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);

    std::ofstream hist;
    if (!cfg.history_path.empty()) {
        hist.open(cfg.history_path.c_str());
        if (hist) {
            hist << "epoch,loss,acc,val_loss,val_acc\n";
        }
    }

    for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        for (size_t start = 0; start < n_train; start += cfg.batch_size) {
            size_t end = std::min(start + static_cast<size_t>(cfg.batch_size), n_train);
            size_t batch_count = end - start;

            std::vector<Layer> grads(layers_.size());
            for (size_t l = 0; l < layers_.size(); ++l) {
                grads[l].W.assign(layers_[l].W.size(),
                                  std::vector<double>(layers_[l].W[0].size(), 0.0));
                grads[l].b.assign(layers_[l].b.size(), 0.0);
            }

            for (size_t bi = start; bi < end; ++bi) {
                size_t idx = indices[bi];

                std::vector< std::vector<double> > A;
                std::vector< std::vector<double> > Z;
                A.push_back(train.x[idx]);

                for (size_t l = 0; l < layers_.size(); ++l) {
                    const Layer& layer = layers_[l];
                    std::vector<double> z(layer.W.size(), 0.0);
                    for (size_t i = 0; i < layer.W.size(); ++i) {
                        double sum = layer.b[i];
                        for (size_t j = 0; j < layer.W[i].size(); ++j) {
                            sum += layer.W[i][j] * A[l][j];
                        }
                        z[i] = sum;
                    }
                    Z.push_back(z);
                    std::vector<double> a;
                    if (l + 1 == layers_.size()) {
                        a = softmax(z);
                    } else {
                        a.resize(z.size());
                        for (size_t i = 0; i < z.size(); ++i) {
                            a[i] = sigmoid(z[i]);
                        }
                    }
                    A.push_back(a);
                }

                std::vector<double> dZ = A.back();
                int y = train.y[idx];
                dZ[y] -= 1.0;

                for (int l = static_cast<int>(layers_.size()) - 1; l >= 0; --l) {
                    const Layer& layer = layers_[l];
                    std::vector<double>& db = grads[l].b;
                    std::vector< std::vector<double> >& dW = grads[l].W;

                    for (size_t i = 0; i < layer.W.size(); ++i) {
                        db[i] += dZ[i];
                        for (size_t j = 0; j < layer.W[i].size(); ++j) {
                            dW[i][j] += dZ[i] * A[l][j];
                        }
                    }

                    if (l > 0) {
                        std::vector<double> dA_prev(layer.W[0].size(), 0.0);
                        for (size_t j = 0; j < layer.W[0].size(); ++j) {
                            double sum = 0.0;
                            for (size_t i = 0; i < layer.W.size(); ++i) {
                                sum += layer.W[i][j] * dZ[i];
                            }
                            dA_prev[j] = sum;
                        }
                        std::vector<double> dZ_prev(dA_prev.size(), 0.0);
                        for (size_t j = 0; j < dA_prev.size(); ++j) {
                            double a = A[l][j];
                            dZ_prev[j] = dA_prev[j] * a * (1.0 - a);
                        }
                        dZ = dZ_prev;
                    }
                }
            }

            double lr = cfg.learning_rate / static_cast<double>(batch_count);
            for (size_t l = 0; l < layers_.size(); ++l) {
                for (size_t i = 0; i < layers_[l].W.size(); ++i) {
                    layers_[l].b[i] -= lr * grads[l].b[i];
                    for (size_t j = 0; j < layers_[l].W[i].size(); ++j) {
                        layers_[l].W[i][j] -= lr * grads[l].W[i][j];
                    }
                }
            }
        }

        double train_loss = 0.0;
        int train_correct = 0;
        for (size_t i = 0; i < train.x.size(); ++i) {
            std::vector<double> out = forward_one(train.x[i]);
            train_loss += cross_entropy(out, train.y[i]);
            int pred = (out[0] > out[1]) ? 0 : 1;
            if (pred == train.y[i]) {
                train_correct++;
            }
        }
        train_loss /= static_cast<double>(train.x.size());
        double train_acc = static_cast<double>(train_correct) / static_cast<double>(train.x.size());

        double valid_loss = 0.0;
        int valid_correct = 0;
        for (size_t i = 0; i < valid.x.size(); ++i) {
            std::vector<double> out = forward_one(valid.x[i]);
            valid_loss += cross_entropy(out, valid.y[i]);
            int pred = (out[0] > out[1]) ? 0 : 1;
            if (pred == valid.y[i]) {
                valid_correct++;
            }
        }
        valid_loss /= static_cast<double>(valid.x.size());
        double valid_acc = static_cast<double>(valid_correct) / static_cast<double>(valid.x.size());

        std::cout << "epoch " << epoch << "/" << cfg.epochs
                  << " - loss: " << train_loss
                  << " - acc: " << train_acc
                  << " - val_loss: " << valid_loss
                  << " - val_acc: " << valid_acc << std::endl;
        if (hist) {
            hist << epoch << "," << train_loss << "," << train_acc << ","
                 << valid_loss << "," << valid_acc << "\n";
        }
    }
}

void MLP::predict() {
    std::cout << "Predict stub" << std::endl;
}

std::vector<double> MLP::predict_proba(const std::vector<double>& x) const {
    return forward_one(x);
}

} // namespace mlp
