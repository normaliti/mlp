#include "utils.hpp"
#include "mlp.hpp"
#include "mlp_io.hpp"
#include "dataset.hpp"

#include <algorithm>
#include <cmath>
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

/* static double binary_cross_entropy(double p, int y) {
    const double eps = 1e-12;
    if (p < eps) p = eps;
    if (p > 1.0 - eps) p = 1.0 - eps;
    return -(y * std::log(p) + (1 - y) * std::log(1.0 - p));
} */

struct ScoredLabel {
    double score;
    int label;
};

static void write_roc_csv(const std::vector<ScoredLabel>& data, const std::string& path) {
    size_t pos = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i].label == 1) {
            pos++;
        }
    }
    size_t neg = data.size() - pos;

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
        double thr = sorted[idx].score;
        while (idx < sorted.size() && sorted[idx].score == thr) {
            if (sorted[idx].label == 1) {
                tp++;
            } else {
                fp++;
            }
            idx++;
        }
        double tpr = (pos == 0) ? 0.0 : static_cast<double>(tp) / static_cast<double>(pos);
        double fpr = (neg == 0) ? 0.0 : static_cast<double>(fp) / static_cast<double>(neg);
        file << thr << "," << fpr << "," << tpr << "\n";
    }
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
    int tp = 0;
    int fp = 0;
    int fn = 0;
    std::vector<ScoredLabel> scored;
    scored.reserve(data.x.size());

    for (size_t i = 0; i < data.x.size(); ++i) {
        std::vector<double> out = model.predict(data.x[i]);
        double p = out.size() > 1 ? out[1] : 0.0;
        int y = data.y[i];
        loss += mlp::binary_cross_entropy(p, y);

        int pred = (p >= 0.5) ? 1 : 0;
        if (pred == y) {
            correct++;
        }
        if (pred == 1 && y == 1) tp++;
        else if (pred == 1 && y == 0) fp++;
        else if (pred == 0 && y == 0) {}
        else fn++;

        scored.push_back(ScoredLabel{p, y});
    }

    loss /= static_cast<double>(data.x.size());
    double acc = static_cast<double>(correct) / static_cast<double>(data.x.size());
    double precision = (tp + fp) == 0 ? 0.0 : static_cast<double>(tp) / static_cast<double>(tp + fp);
    double recall = (tp + fn) == 0 ? 0.0 : static_cast<double>(tp) / static_cast<double>(tp + fn);
    double f1 = (precision + recall) == 0.0 ? 0.0 : 2.0 * precision * recall / (precision + recall);

    std::cout << "Samples: " << data.x.size() << "\n";
    std::cout << "Binary cross-entropy: " << loss << "\n";
    std::cout << "Accuracy: " << acc << "\n";
    std::cout << "Precision: " << precision << "\n";
    std::cout << "Recall: " << recall << "\n";
    std::cout << "F1: " << f1 << "\n";

    std::string roc_csv = "models/roc.csv";
    write_roc_csv(scored, roc_csv);
    std::string cmd = "python3 scripts/roc_plot.py " + roc_csv;
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::cout << "Warning: failed to generate ROC plot (exit code " << rc << ")\n";
    }
    return 0;
}
