#include "utils.hpp"
#include "mlp.hpp"
#include "dataset.hpp"
#include "mlp_io.hpp"

#include <filesystem>
#include <iostream>
#include <string>

static void print_usage(const char* prog) {
    std::cout << "Usage:\n";
    std::cout << "  " << prog << " <train_csv> <valid_csv> [scaler_out] [model_out] [options]\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << prog << " data_training.csv data_validation.csv models/scaler.txt \\\n";
    std::cout << "    models/model.txt --layers 30 24 24 2 --epochs 50 --batch_size 8 --lr 0.01\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --layers <list>       Example: --layers 30 24 24 2\n";
    std::cout << "  --epochs <int>        Default: 50\n";
    std::cout << "  --batch_size <int>    Default: 8\n";
    std::cout << "  --lr <float>          Default: 0.01\n";
}

static void ensure_models_dir() {
    std::error_code ec;
    std::filesystem::create_directories("models", ec);
    if (ec) {
        std::cout << "Warning: could not create models/ directory.\n";
    }
}

int main(int argc, char** argv) {
    utils::print_message("train");

    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string train_path = argv[1];
    std::string valid_path = argv[2];
    std::string scaler_path = "models/scaler.txt";
    std::string model_path = "models/model.txt";

    int argi = 3;
    if (argi < argc && std::string(argv[argi]).rfind("--", 0) != 0) {
        scaler_path = argv[argi];
        ++argi;
    }
    if (argi < argc && std::string(argv[argi]).rfind("--", 0) != 0) {
        model_path = argv[argi];
        ++argi;
    }

    mlp::MLPConfig cfg;
    cfg.layers.clear();
    cfg.epochs = 50;
    cfg.batch_size = 8;
    cfg.learning_rate = 0.01;
    cfg.history_path = "models/history.csv";

    while (argi < argc) {
        std::string key = argv[argi];
        if (key == "--layers") {
            cfg.layers.clear();
            ++argi;
            while (argi < argc && std::string(argv[argi]).rfind("--", 0) != 0) {
                cfg.layers.push_back(std::atoi(argv[argi]));
                ++argi;
            }
        } else if (key == "--epochs" && argi + 1 < argc) {
            cfg.epochs = std::atoi(argv[argi + 1]);
            argi += 2;
        } else if (key == "--batch_size" && argi + 1 < argc) {
            cfg.batch_size = std::atoi(argv[argi + 1]);
            argi += 2;
        } else if (key == "--lr" && argi + 1 < argc) {
            cfg.learning_rate = std::atof(argv[argi + 1]);
            argi += 2;
        } else {
            std::cout << "Unknown or incomplete option: " << key << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (cfg.layers.empty()) {
        cfg.layers.push_back(30);
        cfg.layers.push_back(24);
        cfg.layers.push_back(24);
        cfg.layers.push_back(2);
    }

    if (cfg.layers.size() < 4) {
        std::cout << "Error: need input + at least 2 hidden + output layers.\n";
        return 1;
    }
    if (cfg.layers.front() != 30) {
        std::cout << "Error: input layer must be 30 (number of features).\n";
        return 1;
    }
    if (cfg.layers.back() != 2) {
        std::cout << "Error: output layer must be 2 (softmax classes).\n";
        return 1;
    }
    if (cfg.epochs <= 0 || cfg.batch_size <= 0 || cfg.learning_rate <= 0.0) {
        std::cout << "Error: epochs, batch_size and lr must be positive.\n";
        return 1;
    }

    dataset::Dataset train = dataset::load_dataset(train_path);
    dataset::Dataset valid = dataset::load_dataset(valid_path);

    if (train.x.empty() || valid.x.empty()) {
        std::cout << "Error: failed to load datasets.\n";
        return 1;
    }

    dataset::Scaler s = dataset::fit_minmax(train);
    dataset::apply_minmax(train, s);
    dataset::apply_minmax(valid, s);

    ensure_models_dir();

    if (!dataset::save_scaler(scaler_path, s)) {
        std::cout << "Warning: could not save scaler to " << scaler_path << "\n";
    }

    size_t n_features = train.x[0].size();
    std::cout << "Train samples: " << train.x.size() << ", features: " << n_features << "\n";
    std::cout << "Valid samples: " << valid.x.size() << ", features: " << valid.x[0].size() << "\n";
    std::cout << "Scaler saved to: " << scaler_path << "\n";

    if (!train.x.empty() && !train.x[0].empty()) {
        std::cout << "First train sample (first 5 features): ";
        for (size_t i = 0; i < train.x[0].size() && i < 5; ++i) {
            std::cout << train.x[0][i] << (i + 1 < 5 ? ", " : "");
        }
        std::cout << "\n";
    }

    mlp::MLP model;
    model.train(cfg, train, valid);
    if (!mlp::save_model(model, model_path)) {
        std::cout << "Warning: could not save model to " << model_path << "\n";
    } else {
        std::cout << "Model saved to: " << model_path << "\n";
    }
    return 0;
}
