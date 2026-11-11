#include "mlp.hpp"

namespace mlp {

MLP::Vec MLP::forward(const Vec& x, ForwardCache* cache) const {
    if (layers_.empty() || sizes_.size() < 2 || x.size() != static_cast<size_t>(sizes_.front())) {
        return Vec();
    }

    Vec a = x;
    if (cache != nullptr) {
        cache->A.clear();
        cache->Z.clear();
        cache->A.push_back(x);
    }

    for (size_t l = 0; l < layers_.size(); ++l) {
        const Layer& layer = layers_[l];
        Vec z(layer.W.size(), 0.0);

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
                a[i] = activation_forward(z[i], activation_);
            }
        }

        if (cache != nullptr) {
            cache->Z.push_back(z);
            cache->A.push_back(a);
        }
    }

    return a;
}

} // namespace mlp
