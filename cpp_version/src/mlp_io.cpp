#include "mlp_io.hpp"
#include "mlp.hpp"

#include <fstream>

namespace mlp {

static const char* activation_name(Activation act) {
    switch (act) {
    case Activation::Sigmoid:
        return "sigmoid";
    case Activation::Tanh:
        return "tanh";
    case Activation::Relu:
        return "relu";
    }
    return "sigmoid";
}

static Activation activation_from_name(const std::string& name) {
    if (name == "sigmoid") return Activation::Sigmoid;
    if (name == "tanh") return Activation::Tanh;
    if (name == "relu") return Activation::Relu;
    return Activation::Sigmoid;
}

bool save_model(const MLP& model, const std::string& path) {
    if (model.layers_.empty() || model.sizes_.size() < 2) {
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
    file << "activation " << activation_name(model.activation_) << "\n";

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

    return static_cast<bool>(file);
}

bool load_model(MLP& model, const std::string& path) {
    std::ifstream file(path.c_str());
    if (!file) {
        return false;
    }

    std::string tag;
    size_t n_layers = 0;
    if (!(file >> tag >> n_layers) || tag != "layers" || n_layers < 2) {
        return false;
    }

    std::vector<int> sizes(n_layers, 0);
    for (size_t i = 0; i < n_layers; ++i) {
        if (!(file >> sizes[i]) || sizes[i] <= 0) {
            return false;
        }
    }

    Activation activation = Activation::Sigmoid;
    if (!(file >> tag)) {
        return false;
    }
    if (tag == "activation") {
        std::string name;
        if (!(file >> name)) {
            return false;
        }
        activation = activation_from_name(name);
        if (!(file >> tag)) {
            return false;
        }
    }

    std::vector<MLP::Layer> loaded_layers;
    loaded_layers.reserve(n_layers - 1);

    for (size_t l = 0; l + 1 < n_layers; ++l) {
        if (tag != "layer") {
            return false;
        }

        size_t layer_idx = 0;
        if (!(file >> layer_idx) || layer_idx != l) {
            return false;
        }

        size_t out = 0;
        size_t in = 0;
        if (!(file >> tag >> out >> in) || tag != "W") {
            return false;
        }
        if (out != static_cast<size_t>(sizes[l + 1]) || in != static_cast<size_t>(sizes[l])) {
            return false;
        }

        MLP::Layer layer;
        layer.W.assign(out, std::vector<double>(in, 0.0));
        for (size_t i = 0; i < out; ++i) {
            for (size_t j = 0; j < in; ++j) {
                if (!(file >> layer.W[i][j])) {
                    return false;
                }
            }
        }

        size_t bsz = 0;
        if (!(file >> tag >> bsz) || tag != "b" || bsz != out) {
            return false;
        }
        layer.b.assign(bsz, 0.0);
        for (size_t i = 0; i < bsz; ++i) {
            if (!(file >> layer.b[i])) {
                return false;
            }
        }

        loaded_layers.push_back(layer);
        if (l + 2 < n_layers && !(file >> tag)) {
            return false;
        }
    }

    model.sizes_ = sizes;
    model.activation_ = activation;
    model.layers_ = loaded_layers;
    return true;
}

} // namespace mlp
