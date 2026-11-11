#include "mlp.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace mlp {

MLP::MLP() : activation_(Activation::Sigmoid) {}

double MLP::activation_forward(double z, Activation act) {
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

double MLP::activation_backward(double z, double a, Activation act) {
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

MLP::Vec MLP::softmax(const Vec& z) {
    Vec out(z.size(), 0.0);
    if (z.empty()) {
        return out;
    }

    const double maxv = *std::max_element(z.begin(), z.end());
    double sum = 0.0;
    for (size_t i = 0; i < z.size(); ++i) {
        out[i] = std::exp(z[i] - maxv);
        sum += out[i];
    }

    if (sum <= 0.0) {
        return out;
    }

    for (size_t i = 0; i < out.size(); ++i) {
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

    for (size_t l = 0; l + 1 < sizes.size(); ++l) {
        const int in = sizes[l];
        const int out = sizes[l + 1];

        Layer layer;
        layer.W.assign(out, Vec(in, 0.0));
        layer.b.assign(out, 0.0);

        const double stddev = (activation_ == Activation::Relu)
                                ? std::sqrt(2.0 / static_cast<double>(in))
                                : std::sqrt(1.0 / static_cast<double>(in));
        std::normal_distribution<double> dist(0.0, stddev);

        for (int i = 0; i < out; ++i) {
            for (int j = 0; j < in; ++j) {
                layer.W[i][j] = dist(rng);
            }
        }

        layers_.push_back(layer);
    }
}

MLP::Vec MLP::predict(const Vec& x) const {
    return forward(x, nullptr);
}

} // namespace mlp
