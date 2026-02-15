#include "utils.hpp"
#include "mlp.hpp"
#include "mlp_io.hpp"
#include "dataset.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static void print_usage(const char* prog) {
    std::cout << "Usage:\n";
    std::cout << "  " << prog << " <model_path> <scaler_path> <data_csv>\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << prog << " model.txt scaler.txt data_validation.csv\n";
}

struct ScoredLabel {
    double score;
    int label;
};

static void write_roc_csv(const std::vector<ScoredLabel>& data, const std::string& path) {
    size_t pos = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i].label == 1) {
            ++pos;
        }
    }
    const size_t neg = data.size() - pos;

    std::vector<ScoredLabel> sorted = data;
    std::sort(sorted.begin(), sorted.end(),
              [](const ScoredLabel& a, const ScoredLabel& b) { return a.score > b.score; });

    std::filesystem::create_directories("models");
    std::ofstream file(path.c_str());
    if (!file) {
        std::cout << "Warning: could not write ROC CSV: " << path << "\n";
        return;
    }
    file << "threshold,fpr,tpr\n";

    size_t tp = 0;
    size_t fp = 0;
    size_t idx = 0;
    file << "inf,0,0\n";
    while (idx < sorted.size()) {
        const double thr = sorted[idx].score;
        while (idx < sorted.size() && sorted[idx].score == thr) {
            if (sorted[idx].label == 1) {
                ++tp;
            } else {
                ++fp;
            }
            ++idx;
        }
        const double tpr = (pos == 0) ? 0.0 : static_cast<double>(tp) / static_cast<double>(pos);
        const double fpr = (neg == 0) ? 0.0 : static_cast<double>(fp) / static_cast<double>(neg);
        file << thr << "," << fpr << "," << tpr << "\n";
    }
}

int main(int argc, char** argv) {
    utils::print_message("predict");

    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string scaler_path = argv[2];
    const std::string data_path = argv[3];

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

    if (!dataset::apply_minmax(data, s)) {
        std::cout << "Error: failed to apply min-max scaling (shape mismatch).\n";
        return 1;
    }

    double loss = 0.0;
    int correct = 0;
    int tp = 0;
    int fp = 0;
    int fn = 0;
    std::vector<ScoredLabel> scored;
    scored.reserve(data.x.size());

    for (size_t i = 0; i < data.x.size(); ++i) {
        const std::vector<double> out = model.predict(data.x[i]);
        if (out.size() < 2) {
            std::cout << "Error: model prediction has invalid output size.\n";
            return 1;
        }

        const double p = out[1];
        const int y = data.y[i];
        loss += mlp::binary_cross_entropy(p, y);

        const int pred = (p >= 0.5) ? 1 : 0;
        if (pred == y) {
            ++correct;
        }
        if (pred == 1 && y == 1) ++tp;
        else if (pred == 1 && y == 0) ++fp;
        else if (pred == 0 && y == 1) ++fn;

        scored.push_back(ScoredLabel{p, y});
    }

    loss /= static_cast<double>(data.x.size());
    const double acc = static_cast<double>(correct) / static_cast<double>(data.x.size());
    const double precision = (tp + fp) == 0 ? 0.0 : static_cast<double>(tp) / static_cast<double>(tp + fp);
    const double recall = (tp + fn) == 0 ? 0.0 : static_cast<double>(tp) / static_cast<double>(tp + fn);
    const double f1 = (precision + recall) == 0.0 ? 0.0 : 2.0 * precision * recall / (precision + recall);

    std::cout << "Samples: " << data.x.size() << "\n";
    std::cout << "Binary cross-entropy: " << loss << "\n";
    std::cout << "Accuracy: " << acc << "\n";
    std::cout << "Precision: " << precision << "\n";
    std::cout << "Recall: " << recall << "\n";
    std::cout << "F1: " << f1 << "\n";

    const std::string roc_csv = "models/roc.csv";
    write_roc_csv(scored, roc_csv);
    const int rc = utils::run_python_plot("scripts/roc_plot.py", roc_csv);
    if (rc != 0) {
        std::cout << "Warning: failed to generate ROC plot (exit code " << rc << ")\n";
    }

    return 0;
}
