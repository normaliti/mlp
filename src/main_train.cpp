#include "utils.hpp"
#include "mlp.hpp"
#include "dataset.hpp"
#include "mlp_io.hpp"
#include "train_cli.hpp"

#include <filesystem>
#include <iostream>

static void ensure_models_dir() {
    std::error_code ec;
    std::filesystem::create_directories("models", ec);
    if (ec) {
        std::cout << "Warning: could not create models/ directory.\n";
    }
}

int main(int argc, char** argv) {
    utils::print_message("train");

    train_cli::Options options;
    std::string parse_error;
    if (!train_cli::parse_args(argc, argv, options, parse_error)) {
        if (!parse_error.empty()) {
            std::cout << parse_error << "\n";
        }
        train_cli::print_usage(argv[0]);
        return 1;
    }

    mlp::MLPConfig cfg = options.cfg;

    if (cfg.epochs <= 0 || cfg.batch_size <= 0 || cfg.learning_rate <= 0.0) {
        std::cout << "Error: epochs, batch_size and lr must be positive.\n";
        return 1;
    }

    dataset::Dataset train = dataset::load_dataset(options.train_path);
    dataset::Dataset valid = dataset::load_dataset(options.valid_path);
    if (train.x.empty() || valid.x.empty()) {
        std::cout << "Error: failed to load datasets.\n";
        return 1;
    }

    const int feature_count = static_cast<int>(train.x[0].size());
    if (cfg.layers.empty()) {
        cfg.layers.push_back(feature_count);
        cfg.layers.push_back(24);
        cfg.layers.push_back(24);
        cfg.layers.push_back(2);
    }

    if (cfg.layers.size() < 3) {
        std::cout << "Error: need at least input + hidden + output layers.\n";
        return 1;
    }
    if (cfg.layers.front() != feature_count) {
        std::cout << "Error: input layer must match feature count: " << feature_count << "\n";
        return 1;
    }
    if (cfg.layers.back() != 2) {
        std::cout << "Error: output layer must be 2 (softmax classes).\n";
        return 1;
    }

    dataset::Scaler s = dataset::fit_minmax(train);
    if (s.min.empty() || s.max.empty()) {
        std::cout << "Error: could not fit min-max scaler.\n";
        return 1;
    }
    if (!dataset::apply_minmax(train, s) || !dataset::apply_minmax(valid, s)) {
        std::cout << "Error: failed to apply min-max scaling (shape mismatch).\n";
        return 1;
    }

    ensure_models_dir();

    if (!dataset::save_scaler(options.scaler_path, s)) {
        std::cout << "Warning: could not save scaler to " << options.scaler_path << "\n";
    }

    mlp::MLP model;
    model.train(cfg, train, valid);
    if (!mlp::save_model(model, options.model_path)) {
        std::cout << "Warning: could not save model to " << options.model_path << "\n";
    } else {
        std::cout << "Model saved to: " << options.model_path << "\n";
    }

    if (options.do_plot) {
        const int rc = utils::run_python_plot("scripts/plot.py", cfg.history_path);
        if (rc != 0) {
            std::cout << "Warning: failed to generate plots (exit code " << rc << ")\n";
        }
    }

    return 0;
}
