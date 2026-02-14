#include "mlp.hpp"
#include "dataset.hpp"
#include <fstream>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <algorithm>

namespace mlp {

MLP::MLP() : activation_(Activation::Sigmoid) {}

static double activation_forward(double z, Activation act) {
    switch (act) {
    case Activation::Sigmoid:
        return 1.0 / (1.0 + std::exp(-z));
    case Activation::Tanh:
        return std::tanh(z);
    case Activation::Relu:
        return z > 0.0 ? z : 0.0;
    }
    return 1.0 / (1.0 + std::exp(-z));
}

static double activation_backward(double z, double a, Activation act) {
    (void)z;
    switch (act) {
    case Activation::Sigmoid:
        return a * (1.0 - a);
    case Activation::Tanh:
        return 1.0 - a * a;
    case Activation::Relu:
        return z > 0.0 ? 1.0 : 0.0;
    }
    return a * (1.0 - a);
}

static std::vector<double> softmax(const std::vector<double>& z) {
    std::vector<double> out(z.size(), 0.0);
    if (z.empty()) {
        return out;
    }
    double maxv = *std::max_element(z.begin(), z.end());

    double sum = 0.0;
    for (size_t i = 0; i < z.size(); i++) {
        out[i] = std::exp(z[i] - maxv);
        sum += out[i];
    }
    if (sum == 0.0) {
        return out;
    }
    for (size_t i = 0; i < out.size(); i++) {
        out[i] /= sum;
    }
    return out;
}

double binary_cross_entropy(double p, int y) {
    const double eps = 1e-12;
    if (p < eps) p = eps;
    if (p > 1.0 - eps) p = 1.0 - eps;
    return -(y * std::log(p) + (1 - y) * std::log(1.0 - p));
}

void MLP::init_weights(const std::vector<int>& sizes, unsigned int seed) {
    layers_.clear();
    sizes_ = sizes;
    std::mt19937 rng(seed);

    for (size_t l = 0; l < sizes.size() - 1; l++) {
        int in = sizes[l];
        int out = sizes[l + 1];
        Layer layer;
        layer.W.assign(out, std::vector<double>(in, 0.0));
        layer.b.assign(out, 0.0);
        double stddev = 0.0;
        if (activation_ == Activation::Relu) 
            stddev = stddev = std::sqrt(2.0 / static_cast<double>(in));        
        else              
            //stddev = std::sqrt(2 / static_cast<double>(in + out));        
            stddev = std::sqrt(1 / static_cast<double>(in));
        std::normal_distribution<double> dist(0.0, stddev);
        
        for (int i = 0; i < out; i++)
        {
            for (int j = 0; j < in; j++) 
            {
                layer.W[i][j] = dist(rng);
            }
            //layer.b[i] = dist(rng);
            layer.b[i] = 0;
        }
        layers_.push_back(layer);
    }
}

std::vector<double> MLP::feedforward(const std::vector<double>& x) const 
{
    std::vector<double> a = x;
    double sum;
    for (size_t l = 0; l < layers_.size(); l++) // layers_.size = 3
    { 
        const Layer& layer = layers_[l]; // current layer 0 = W and b, W has a size (24*30)
        std::vector<double> z(layer.W.size(), 0.0);// out values in first hidden layer vector has size = 24
        for (size_t i = 0; i < layer.W.size(); i++) 
        {
            sum = layer.b[i];
            for (size_t j = 0; j < layer.W[i].size(); j++) 
            {
                sum += layer.W[i][j] * a[j];
            }
            z[i] = sum;
        }
        if (l + 1 == layers_.size()) 
            a = softmax(z);
         
        else 
        {
            a.resize(z.size());
            for (size_t i = 0; i < z.size(); i++) 
                a[i] = activation_forward(z[i], activation_);            
        }
    }
    return a;
}

void MLP::feedforward_with_cache(const std::vector<double>& x,
                                 std::vector< std::vector<double> >& A,
                                 std::vector< std::vector<double> >& Z) const {
    A.clear();
    Z.clear();
    A.push_back(x);

    for (size_t l = 0; l < layers_.size(); l++) {
        const Layer& layer = layers_[l];
        
        std::vector<double> z(layer.W.size(), 0.0);
        for (size_t i = 0; i < layer.W.size(); i++) {
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
                a[i] = activation_forward(z[i], activation_);
            }
        }
        A.push_back(a);
    }
}

void MLP::backprop(const std::vector< std::vector<double> >& A,
                   const std::vector< std::vector<double> >& Z,
                   int y,
                   std::vector<Layer>& grads) const {
    (void)Z;
    std::vector<double> dZ = A.back();
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
                double z = Z[l - 1][j];
                dZ_prev[j] = dA_prev[j] * activation_backward(z, a, activation_);
            }
            dZ = dZ_prev;
        }
    }
}

void MLP::apply_grads(const std::vector<Layer>& grads, double lr) {
    for (size_t l = 0; l < layers_.size(); ++l) {
        for (size_t i = 0; i < layers_[l].W.size(); ++i) {
            layers_[l].b[i] -= lr * grads[l].b[i];
            for (size_t j = 0; j < layers_[l].W[i].size(); ++j) {
                layers_[l].W[i][j] -= lr * grads[l].W[i][j];
            }
        }
    }
}

MLP::Metrics MLP::evaluate(const dataset::Dataset& data) const {
    double loss = 0.0;
    int correct = 0;
    for (size_t i = 0; i < data.x.size(); ++i) {
        std::vector<double> out = feedforward(data.x[i]);
        double p = out.size() > 1 ? out[1] : 0.0;
        loss += binary_cross_entropy(p, data.y[i]);
        int pred = (out[0] > out[1]) ? 0 : 1;
        if (pred == data.y[i]) {
            correct++;
        }
    }
    loss /= static_cast<double>(data.x.size());
    double acc = static_cast<double>(correct) / static_cast<double>(data.x.size());
    Metrics m;
    m.loss = loss;
    m.acc = acc;
    return m;
}

std::vector<double> MLP::predict(const std::vector<double>& x) const {
    return feedforward(x);
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

    activation_ = cfg.activation;
    init_weights(cfg.layers, 42);

    std::ofstream history_file(cfg.history_path.c_str());
    if (history_file) {
        history_file << "epoch,loss,acc,val_loss,val_acc\n";
    } else {
        std::cout << "Warning: could not open history file: " << cfg.history_path << "\n";
    }

    size_t n_train = train.x.size();
    std::vector<size_t> indices(n_train);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);

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
                feedforward_with_cache(train.x[idx], A, Z);
                backprop(A, Z, train.y[idx], grads);
            }

            double lr = cfg.learning_rate / static_cast<double>(batch_count);
            apply_grads(grads, lr);
        }

        Metrics train_metrics = evaluate(train);
        Metrics valid_metrics = evaluate(valid);

        std::cout << "epoch " << epoch << "/" << cfg.epochs
                  << " - loss: " << train_metrics.loss
                  << " - acc: " << train_metrics.acc
                  << " - val_loss: " << valid_metrics.loss
                  << " - val_acc: " << valid_metrics.acc << std::endl;

        if (history_file) {
            history_file << epoch << "," << train_metrics.loss << "," << train_metrics.acc << ","
                         << valid_metrics.loss << "," << valid_metrics.acc << "\n";
        }
    }
}



} // namespace mlp
