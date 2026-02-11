#include "utils.hpp"
#include "mlp.hpp"
#include "mlp_io.hpp"
#include "dataset.hpp"

#include <cmath>
#include <iostream>
#include <string>

static void print_usage(const char* prog) {
    std::cout << "Usage:\n";
    std::cout << "  " << prog << " <model_path> <scaler_path> <data_csv>\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << prog << " model.txt scaler.txt data_validation.csv\n";
}

static double cross_entropy(const std::vector<double>& probs, int y) {
    const double eps = 1e-12;
    double p = probs[y];
    if (p < eps) p = eps;
    return -std::log(p);
}

int main(int argc, char** argv) {
    utils::print_message("predict");

    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string scaler_path = argv[2];
    std::string data_path = argv[3];

    mlp::MLP model;
    if (!mlp::load_model(model, model_path)) {
        std::cout << "Error: failed to load model from " << model_path << "\n";
        return 1;
    }

    dataset::Scaler s;
    if (!dataset::load_scaler(scaler_path, s)) {
        std::cout << "Error: failed to load scaler from " << scaler_path << "\n";
        return 1;
    }

    dataset::Dataset data = dataset::load_dataset(data_path);
    if (data.x.empty()) {
        std::cout << "Error: failed to load dataset.\n";
        return 1;
    }

    dataset::apply_minmax(data, s);

    double loss = 0.0;
    int correct = 0;
    for (size_t i = 0; i < data.x.size(); ++i) {
        std::vector<double> out = model.predict_proba(data.x[i]);
        loss += cross_entropy(out, data.y[i]);
        int pred = (out[0] > out[1]) ? 0 : 1;
        if (pred == data.y[i]) {
            correct++;
        }
    }
    loss /= static_cast<double>(data.x.size());
    double acc = static_cast<double>(correct) / static_cast<double>(data.x.size());

    std::cout << "Samples: " << data.x.size() << "\n";
    std::cout << "Loss: " << loss << "\n";
    std::cout << "Accuracy: " << acc << "\n";
    return 0;
}
