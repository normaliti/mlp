#ifndef MLP_HPP
#define MLP_HPP

#include <string>
#include <vector>

namespace dataset {
struct Dataset;
}

namespace mlp {

struct MLPConfig {
    std::vector<int> layers;
    int epochs;
    int batch_size;
    double learning_rate;
    std::string history_path;
};

class MLP {
public:
    MLP();
    void train(const MLPConfig& cfg, const dataset::Dataset& train, const dataset::Dataset& valid);
    void predict();
    std::vector<double> predict_proba(const std::vector<double>& x) const;

private:
    friend bool save_model(const MLP& model, const std::string& path);
    friend bool load_model(MLP& model, const std::string& path);

    struct Layer {
        std::vector< std::vector<double> > W;
        std::vector<double> b;
    };

    std::vector<Layer> layers_;
    std::vector<int> sizes_;

    void init_weights(const std::vector<int>& sizes, unsigned int seed);
    std::vector<double> forward_one(const std::vector<double>& x) const;
};

} // namespace mlp

#endif
