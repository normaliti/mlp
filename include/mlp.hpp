#ifndef MLP_HPP
#define MLP_HPP

#include <string>
#include <vector>

namespace dataset {
struct Dataset;
}

namespace mlp {

enum class Activation {
    Sigmoid,
    Tanh,
    Relu
};

struct MLPConfig {
    std::vector<int> layers;
    int epochs;
    int batch_size;
    double learning_rate;
    std::string history_path;
    Activation activation;
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

    struct Metrics {
        double loss;
        double acc;
    };

    struct Layer {
        std::vector< std::vector<double> > W;
        std::vector<double> b;
    };

    std::vector<Layer> layers_;
    std::vector<int> sizes_;
    Activation activation_;

    void init_weights(const std::vector<int>& sizes, unsigned int seed);
    std::vector<double> feedforward(const std::vector<double>& x) const;
    void feedforward_with_cache(const std::vector<double>& x,
                                std::vector< std::vector<double> >& A,
                                std::vector< std::vector<double> >& Z) const;
    void backprop(const std::vector< std::vector<double> >& A,
                  const std::vector< std::vector<double> >& Z,
                  int y,
                  std::vector<Layer>& grads) const;
    void apply_grads(const std::vector<Layer>& grads, double lr);
    Metrics evaluate(const dataset::Dataset& data) const;
};

} // namespace mlp

#endif
