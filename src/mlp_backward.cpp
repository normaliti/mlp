#include "mlp.hpp"

namespace mlp {

void MLP::backprop(const std::vector<Vec>& A,
                   const std::vector<Vec>& Z,
                   int y,
                   std::vector<Layer>& grads) const {
    if (A.empty() || Z.size() != layers_.size() || A.size() != layers_.size() + 1) {
        return;
    }

    Vec dZ = A.back();
    if (y < 0 || y >= static_cast<int>(dZ.size())) {
        return;
    }
    dZ[static_cast<size_t>(y)] -= 1.0;

    for (int l = static_cast<int>(layers_.size()) - 1; l >= 0; --l) {
        const Layer& layer = layers_[static_cast<size_t>(l)];
        Vec& db = grads[static_cast<size_t>(l)].b;
        Mat& dW = grads[static_cast<size_t>(l)].W;

        for (size_t i = 0; i < layer.W.size(); ++i) {
            db[i] += dZ[i];
            for (size_t j = 0; j < layer.W[i].size(); ++j) {
                dW[i][j] += dZ[i] * A[static_cast<size_t>(l)][j];
            }
        }

        if (l > 0) {
            Vec dA_prev(layer.W[0].size(), 0.0);
            for (size_t j = 0; j < layer.W[0].size(); ++j) {
                double sum = 0.0;
                for (size_t i = 0; i < layer.W.size(); ++i) {
                    sum += layer.W[i][j] * dZ[i];
                }
                dA_prev[j] = sum;
            }

            Vec dZ_prev(dA_prev.size(), 0.0);
            for (size_t j = 0; j < dA_prev.size(); ++j) {
                const double a = A[static_cast<size_t>(l)][j];
                const double z = Z[static_cast<size_t>(l - 1)][j];
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

} // namespace mlp
