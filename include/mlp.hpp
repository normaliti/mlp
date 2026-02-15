#ifndef MLP_HPP
#define MLP_HPP

#include <iosfwd>
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
    unsigned int seed;
    std::string history_path;
    Activation activation;
};

class MLP {
public:
    using Vec = std::vector<double>;
    using Mat = std::vector< Vec >;

    MLP();
    void train(const MLPConfig& cfg, const dataset::Dataset& train, const dataset::Dataset& valid);
    Vec predict(const Vec& x) const;

private:
    friend bool save_model(const MLP& model, const std::string& path);
    friend bool load_model(MLP& model, const std::string& path);

    struct Metrics {
        double loss;
        double acc;
    };

    struct Layer {
        Mat W;
        Vec b;
    };

    struct ForwardCache {
        std::vector<Vec> A;
        std::vector<Vec> Z;
    };

    std::vector<Layer> layers_;
    std::vector<int> sizes_;
    Activation activation_;

    static double activation_forward(double z, Activation act);
    static double activation_backward(double z, double a, Activation act);
    static Vec softmax(const Vec& z);

    void init_weights(const std::vector<int>& sizes, unsigned int seed);
    Vec forward(const Vec& x, ForwardCache* cache) const;
    void backprop(const std::vector<Vec>& A,
                  const std::vector<Vec>& Z,
                  int y,
                  std::vector<Layer>& grads) const;
    void apply_grads(const std::vector<Layer>& grads, double lr);
    Metrics evaluate(const dataset::Dataset& data) const;
};

double binary_cross_entropy(double p, int y);

} // namespace mlp

#endif
