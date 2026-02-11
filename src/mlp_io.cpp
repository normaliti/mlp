#include "mlp_io.hpp"
#include "mlp.hpp"

#include <fstream>

namespace mlp {

bool save_model(const MLP& model, const std::string& path) {
    if (model.layers_.empty() || model.sizes_.empty()) {
        return false;
    }
    std::ofstream file(path.c_str());
    if (!file) {
        return false;
    }
    file << "layers " << model.sizes_.size() << "\n";
    for (size_t i = 0; i < model.sizes_.size(); ++i) {
        file << model.sizes_[i] << (i + 1 < model.sizes_.size() ? " " : "");
    }
    file << "\n";
    for (size_t l = 0; l < model.layers_.size(); ++l) {
        const MLP::Layer& layer = model.layers_[l];
        file << "layer " << l << "\n";
        file << "W " << layer.W.size() << " " << (layer.W.empty() ? 0 : layer.W[0].size()) << "\n";
        for (size_t i = 0; i < layer.W.size(); ++i) {
            for (size_t j = 0; j < layer.W[i].size(); ++j) {
                file << layer.W[i][j] << (j + 1 < layer.W[i].size() ? " " : "");
            }
            file << "\n";
        }
        file << "b " << layer.b.size() << "\n";
        for (size_t i = 0; i < layer.b.size(); ++i) {
            file << layer.b[i] << (i + 1 < layer.b.size() ? " " : "");
        }
        file << "\n";
    }
    return true;
}

bool load_model(MLP& model, const std::string& path) {
    std::ifstream file(path.c_str());
    if (!file) {
        return false;
    }
    std::string tag;
    size_t n_layers = 0;
    file >> tag >> n_layers;
    if (tag != "layers" || n_layers < 2) {
        return false;
    }
    model.sizes_.assign(n_layers, 0);
    for (size_t i = 0; i < n_layers; ++i) {
        file >> model.sizes_[i];
    }
    model.layers_.clear();
    for (size_t l = 0; l + 1 < n_layers; ++l) {
        size_t layer_idx = 0;
        file >> tag >> layer_idx;
        if (tag != "layer") {
            return false;
        }
        MLP::Layer layer;
        size_t out = 0;
        size_t in = 0;
        file >> tag >> out >> in;
        if (tag != "W") {
            return false;
        }
        layer.W.assign(out, std::vector<double>(in, 0.0));
        for (size_t i = 0; i < out; ++i) {
            for (size_t j = 0; j < in; ++j) {
                file >> layer.W[i][j];
            }
        }
        size_t bsz = 0;
        file >> tag >> bsz;
        if (tag != "b") {
            return false;
        }
        layer.b.assign(bsz, 0.0);
        for (size_t i = 0; i < bsz; ++i) {
            file >> layer.b[i];
        }
        model.layers_.push_back(layer);
    }
    return true;
}

} // namespace mlp
